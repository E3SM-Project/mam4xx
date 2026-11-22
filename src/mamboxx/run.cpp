#include "mam4xx/aero_config.hpp"
#include "mam4xx/aero_modes.hpp"
#include "mam4xx/aero_species.hpp"
#include "mamboxx.hpp"
#include <mam4xx/mam4.hpp>
#include <mam4xx/atmosphere.hpp>

#include <ekat_assert.hpp>
#include <ekat_team_policy_utils.hpp>
#include <string>

// Unresolved questions/issues:
// 1. How do we accommodate different numbers of vertical levels? mam4::nlev
//    is currently a compile-time parameter.
// 2. We need an AtmosphereState instead of an immutable AtmosphereConfig, in
//    which water vapor/ice/etc are tracked and evolved. This means additional
//    input data in the YAML file. See the DryAtmosphere struct in mam_coupling.hpp
//    for the essentials.
// 3. How do we handle emissions and external forcing? Use input from the YAML file?

namespace mamboxx {

namespace {

using View1D            = mam4::DeviceType::view_1d<Real>;
using View2D            = mam4::DeviceType::view_2d<Real>;
using ThreadTeam        = mam4::ThreadTeam;
using TeamPolicyFactory = ekat::TeamPolicyFactory<ekat::KokkosTypes<ekat::DefaultDevice>::ExeSpace>;

using Microphysics = mam4::microphysics::Microphysics;

struct Processes {
  std::unique_ptr<Microphysics> microphysics;
};

struct State {
  mam4::Atmosphere  atmosphere;
  mam4::Prognostics prognostics;
  mam4::Diagnostics diagnostics;

  struct {
    mam4::microphysics::ColumnLocation     location;
    mam4::microphysics::ExternalConditions external_conditions;
    mam4::microphysics::FixedSpeciesView   invariants;
  } microphysics;
};

template <typename T>
std::vector<std::string>
gather_names_from_mapped_ids(const std::map<std::string, T> &indices_for_names) {
  std::vector<std::string> names(indices_for_names.size());
  for (auto iter = mode_indices_for_names.begin(); iter != mode_indices_for_names.end(); ++iter) {
    names[int(iter->second)] = iter->first;
  }
  return std::move(names);
}

void copy_vector_to_view(mam4::ColumnView device_view, const std::vector<Real> &vec, HostColumnView host_view) {
  for (int k = 0; k < vec.size(); ++k) {
    host_view[k] = vec[k];
  }
  Kokkos::deep_copy(device_view, host_view);
}

void copy_vector_to_view(mam4::ConstColumnView device_view, const std::vector<Real> &vec, HostColumnView host_view) {
  for (int k = 0; k < vec.size(); ++k) {
    host_view[k] = vec[k];
  }
  mam4::ColumnView non_const_view((Real*)device_view.data(), mam4::nlev);
  Kokkos::deep_copy(non_const_view, host_view);
}

void copy_view_to_vector(mam4::ColumnView device_view, std::vector<Real> &vec, HostColumnView host_view) {
  Kokkos::deep_copy(host_view, device_view);
  vec.resize(host_view.size());
  for (int k = 0; k < vec.size(); ++k) {
    vec[k] = host_view[k];
  }
}

std::tuple<Processes, State> initialize(const Config &config) {
  int nlev = config.atmosphere.temperature.size();
  EKAT_REQUIRE(nlev == mam4::nlev,
      "requested number of vertical levels (" << nlev <<
      ") differs from compile-time value (" << mam4::nlev <<
      ") -- reconfigure with MAM4XX_NUM_VERTICALS");

  Processes processes = {};
  processes.microphysics = std::make_unique<Microphysics>(config.control.microphysics);

  State state = {};
  state.atmosphere = mam4::Atmosphere(nlev, 
    create_column_view(nlev), // temperature
    create_column_view(nlev), // pressure
    create_column_view(nlev), // qv
    create_column_view(nlev), // qc
    create_column_view(nlev), // nqc
    create_column_view(nlev), // qi
    create_column_view(nlev), // nqi
    create_column_view(nlev), // z
    create_column_view(nlev), // hdp
    create_column_view(nlev), // intp
    create_column_view(nlev), // cf
    create_column_view(nlev), // w
    config.atmosphere.pblh);

  state.prognostics = mam4::Prognostics(nlev);
  for (int m = 0; m < mam4::AeroConfig::num_modes(); ++m) {
    state.prognostics.n_mode_i[m] = create_column_view(nlev);
    state.prognostics.n_mode_c[m] = create_column_view(nlev);
    for (int a = 0; a < mam4::AeroConfig::num_aerosol_ids(); ++a) {
      state.prognostics.q_aero_i[m][a] = create_column_view(nlev);
      state.prognostics.q_aero_c[m][a] = create_column_view(nlev);
      state.prognostics.uptkaer[m][a]  = create_column_view(nlev);
    }
    for (int g = 0; g < mam4::AeroConfig::num_gas_ids(); ++g) {
      state.prognostics.q_gas[g] = create_column_view(nlev);
      state.prognostics.q_gas_avg[g] = create_column_view(nlev);
    }
  }

  auto mode_names         = gather_names_from_mapped_ids(mode_indices_for_names);
  auto aero_species_names = gather_names_from_mapped_ids(aero_ids_for_names);
  auto gas_species_names  = gather_names_from_mapped_ids(gas_ids_for_names);

  // copy data to the device
  auto host_view = create_host_column_view(nlev);
  copy_vector_to_view(state.atmosphere.temperature, config.atmosphere.temperature, host_view);
  copy_vector_to_view(state.atmosphere.pressure, config.atmosphere.pressure, host_view);
  copy_vector_to_view(state.atmosphere.vapor_mixing_ratio, config.atmosphere.vapor_mixing_ratio, host_view);
  // FIXME: etc
  for (int m = 0; m < mam4::AeroConfig::num_modes(); ++m) {
    auto m_iter = config.aerosols.modes.find(mode_names[m]);
    EKAT_REQUIRE(m_iter != config.aerosols.modes.end(),
        "Missing aerosol mode: " << mode_names[m]);
    copy_vector_to_view(state.prognostics.n_mode_i[m], m_iter->second.number_mixing_ratio, host_view);
    for (int s = 0; s < 7; ++s) {
      auto aero_id = mam4::mode_aero_species(m, s);
      auto aero_name = aero_species_names[int(aero_id)];
      auto s_iter = m_iter->second.mass_mixing_ratios.find(aero_name);
      EKAT_REQUIRE(s_iter != m_iter->second.mass_mixing_ratios.end(),
          "Missing aerosol species in mode " << mode_names[m] << ": " << aero_name);
      copy_vector_to_view(state.prognostics.q_aero_i[m][s], s_iter->second, host_view);
      // FIXME: what about cloudborne aerosols?
    }
    for (int g = 0; g < int(mam4::GasId::NumSpecies); ++g) {
      auto gas_name = gas_species_names[g];
      auto g_iter = config.gases.mass_mixing_ratios.find(gas_name);
      if (g_iter != config.gases.mass_mixing_ratios.end()) {
        copy_vector_to_view(state.prognostics.q_gas[g], g_iter->second, host_view);
      }
    }
  }
  return std::make_tuple(std::move(processes), state);
}

void step(const Processes &processes, State &state, Real dt) {
  if (processes.microphysics.get() != nullptr) {
    processes.microphysics->step(state.atmosphere, state.prognostics,
                                 state.microphysics.location, state.microphysics.external_conditions,
                                 state.microphysics.invariants, dt);
  }
}

void update_diagnostics(const Processes &processes, const State &state, Real dt, Diagnostics &diagnostics) {
  mam4::Diagnostics diags = {};
  if (processes.microphysics.get() != nullptr) {
    processes.microphysics->compute_diagnostics(state.atmosphere, state.prognostics, diags);
  }
}

//-----------------------------------------------------------------
// Construction of an aerosol state for use in verification checks
//-----------------------------------------------------------------

PyObject *create_python_column(mam4::ColumnView device_view, HostColumnView host_view) {
  PyObject *py_col = PyList_New(mam4::nlev);
  Kokkos::deep_copy(host_view, device_view);
  for (auto i = 0; i < mam4::nlev; ++i) {
    PyList_SetItem(py_col, i, PyFloat_FromDouble((double)host_view(i)));
  }
  return py_col;
}

PyObject *create_python_atmosphere(const mam4::Atmosphere &atmosphere, HostColumnView host_view) {
  PyObject *py_atm = _PyObject_New(&PyBaseObject_Type);
  return py_atm;
}

PyObject *create_python_mode(const mam4::Prognostics &prognostics, mam4::ModeIndex mode_index,
                             HostColumnView host_view) {
  PyObject *py_mode = _PyObject_New(&PyBaseObject_Type);
  int m = int(mode_index);
  switch (mode_index) {
    case mam4::ModeIndex::Accumulation:
      break;
    case mam4::ModeIndex::Aitken:
      break;
    case mam4::ModeIndex::Coarse:
      break;
    case mam4::ModeIndex::PrimaryCarbon:
      break;
    default:
      EKAT_REQUIRE(false, "Invalid aerosol mode index: " << m);
  };
  PyObject *py_interstitial = _PyObject_New(&PyBaseObject_Type);
  PyObject *py_n_i = create_python_column(prognostics.n_mode_i[m], host_view);
  PyObject_SetAttrString(py_interstitial, "number_density", py_n_i);
  int num_species = mam4::AeroConfig::nspec_amode(m);
  for (int s = 0; s < num_species; ++s) {
    mam4::AeroId a_id = mam4::mode_aero_species(m, s);
    PyObject *py_q_i = create_python_column(prognostics.q_aero_i[m][s], host_view);
    std::string species_name = mam4::aero_id_str(a_id).c_str();
    PyObject_SetAttrString(py_interstitial, species_name.c_str(), py_q_i);
  }
  PyObject_SetAttrString(py_mode, "interstitial", py_interstitial);

  PyObject *py_cloudborne = _PyObject_New(&PyBaseObject_Type);
  PyObject *py_n_c = create_python_column(prognostics.n_mode_c[m], host_view);
  PyObject_SetAttrString(py_cloudborne, "number_density", py_n_c);
  for (int s = 0; s < num_species; ++s) {
    mam4::AeroId a_id = mam4::mode_aero_species(m, s);
    PyObject *py_q_i = create_python_column(prognostics.q_aero_c[m][s], host_view);
    std::string species_name = mam4::aero_id_str(a_id).c_str();
    PyObject_SetAttrString(py_cloudborne, species_name.c_str(), py_q_i);
  }
  PyObject_SetAttrString(py_mode, "cloudborne", py_cloudborne);

  return py_mode;
}

PyObject *create_python_gases(const mam4::Prognostics &prognostics, HostColumnView host_view) {
  PyObject *py_gases = _PyObject_New(&PyBaseObject_Type);
  constexpr Py_ssize_t num_gases = mam4::AeroConfig::num_gas_ids();
  for (Py_ssize_t g = 0; g < num_gases; ++g) {
    static std::string gas_names[num_gases] = {
      "o3", "h2o2", "h2so4", "so2", "dms", "soag", // FIXME: ... right???
    };
    PyObject *py_gas = create_python_column(prognostics.q_gas[g], host_view);
    PyObject_SetAttrString(py_gases, gas_names[g].c_str(), py_gas);
  }
  return py_gases;
}

PyObject *create_python_prognostics(const mam4::Prognostics &prognostics, HostColumnView host_view) {
  PyObject *py_progs = _PyObject_New(&PyBaseObject_Type);
  constexpr Py_ssize_t num_modes = mam4::AeroConfig::num_modes();
  PyObject *modes = _PyObject_New(&PyBaseObject_Type);
  for (Py_ssize_t m = 0; m < num_modes; ++m) {
    static std::string mode_names[num_modes] = {
      "accumulation", "aitken", "coarse", "primary carbon",
    };
    auto mode_index = (mam4::ModeIndex)m;
    PyObject_SetAttrString(modes, mode_names[m].c_str(),
                           create_python_mode(prognostics, mode_index, host_view));
  }
  PyObject_SetAttrString(py_progs, "modes", modes);
  PyObject_SetAttrString(py_progs, "gases", create_python_gases(prognostics, host_view));
  return py_progs;
}

PyObject *create_python_diagnostics(const mam4::Diagnostics &diagnostics, HostColumnView host_view) {
  PyObject *py_diags = _PyObject_New(&PyBaseObject_Type);
  return py_diags;
}

PyObject *create_python_state(const State &state) {
  auto host_view = create_host_column_view(mam4::nlev);
  PyObject *py_state = _PyObject_New(&PyBaseObject_Type);
  PyObject_SetAttrString(py_state, "atmosphere", create_python_atmosphere(state.atmosphere, host_view));
  PyObject_SetAttrString(py_state, "prognostics", create_python_prognostics(state.prognostics, host_view));
  PyObject_SetAttrString(py_state, "diagnostics", create_python_diagnostics(state.diagnostics, host_view));
  return py_state;
}

void finalize(const State &state, Result &result) {
  auto host_view = create_host_column_view(mam4::nlev);

  // copy data out of the state
  for (auto iter = mode_indices_for_names.begin(); iter != mode_indices_for_names.end(); ++iter) {
    std::string     mode_name = iter->first;
    mam4::ModeIndex mode_index = iter->second;
    copy_view_to_vector(state.prognostics.n_mode_i[int(mode_index)],
                        result.aerosols.modes[mode_name].number_mixing_ratio,
                        host_view);
    // FIXME: what about cloudborne aerosols?
    for (auto iter = aero_ids_for_names.begin(); iter != aero_ids_for_names.end(); ++iter) {
      std::string  aero_name = iter->first;
      mam4::AeroId aero_id = iter->second;
      copy_view_to_vector(state.prognostics.q_aero_i[int(mode_index)][int(aero_id)],
                          result.aerosols.modes[mode_name].mass_mixing_ratios[aero_name],
                          host_view);
      // FIXME: what about cloudborne aerosols?
    }
    for (auto iter = gas_ids_for_names.begin(); iter != gas_ids_for_names.end(); ++iter) {
      std::string gas_name = iter->first;
      mam4::GasId gas_id   = iter->second;
      copy_view_to_vector(state.prognostics.q_gas[int(gas_id)],
                          result.gases.mass_mixing_ratios[gas_name],
                          host_view);
    }
  }
}

} // anonymous namespace

Result run(const Config &config) {
  Diagnostics diagnostics = {};

  auto [processes, state] = initialize(config);

  for (int i = 0; i < config.time.nstep; ++i) {
    step(processes, state, config.time.dt);
    update_diagnostics(processes, state, config.time.dt, diagnostics);
  }

  // assemble a Python-sensible aerosol state and run checks on it
  PyObject *py_state = create_python_state(state);
  for (auto check: config.verification.checks) {
    PyObject_Call(check, py_state, NULL);
  }

  Result result = {};
  finalize(state, result);
  return result;
}

}
