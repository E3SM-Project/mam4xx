#include "mamboxx.hpp"
#include <mam4xx/mam4.hpp>
#include <mam4xx/atmosphere.hpp>

#include <ekat_assert.hpp>
#include <ekat_team_policy_utils.hpp>

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

struct State {
  mam4::Atmosphere  atmosphere;
  mam4::Prognostics prognostics;

  View2D invariants;
};

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
  Kokkos::deep_copy(device_view, host_view);
}

void copy_view_to_vector(mam4::ColumnView device_view, std::vector<Real> &vec, HostColumnView host_view) {
  Kokkos::deep_copy(host_view, device_view);
  vec.resize(host_view.size());
  for (int k = 0; k < vec.size(); ++k) {
    vec[k] = host_view[k];
  }
}

void initialize(const Config &config, State &state) {
  int nlev = config.atmosphere.temperature.size();
  EKAT_REQUIRE(nlev == mam4::nlev,
      "requested number of vertical levels (" << nlev <<
      ") differs from compile-time value (" << mam4::nlev <<
      ") -- reconfigure with MAM4XX_NUM_VERTICALS");

  // allocate resources
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

  std::vector<std::string> mode_names(mode_indices_for_names.size());
  for (auto iter = mode_indices_for_names.begin(); iter != mode_indices_for_names.end(); ++iter) {
    mode_names[int(iter->second)] = iter->first;
  }
  std::vector<std::string> aero_species_names(aero_ids_for_names.size());
  for (auto iter = aero_ids_for_names.begin(); iter != aero_ids_for_names.end(); ++iter) {
    aero_species_names[int(iter->second)] = iter->first;
  }
  std::vector<std::string> gas_species_names(gas_ids_for_names.size());
  for (auto iter = gas_ids_for_names.begin(); iter != gas_ids_for_names.end(); ++iter) {
    gas_species_names[int(iter->second)] = iter->first;
  }

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
}

void external_forcing(State &state, Real dt) {
  // TODO: add when appropriate
}

void set_invariants(State &state, Real dt) {
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);
  View1D cnst_offline[mam4::mo_setinv::num_tracer_cnst];
  // FIXME: populate offline constituents 
  Kokkos::parallel_for("mamboxx::set_invariants", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    mam4::mo_setinv::setinv(team,                                    // in
                            state.invariants,                        // out
                            state.atmosphere.temperature,            // in
                            state.atmosphere.vapor_mixing_ratio,     // in
                            cnst_offline,                            // in
                            state.atmosphere.pressure);              // in
  });
}

void compute_o3_column_density(State &state, Real dt) {
}

void photo_table(State &state, Real dt) {
}

void sethet(State &state, Real dt) {
}

void drydep_xactive(State &state, Real dt) {
}

void gas_phase_chemistry(State &state, Real dt) {
}

void setsox(State &state, Real dt) {
}

void modal_aero_amicphys_intr(State &state, Real dt) {
}

void vmr2mmr_cw(State &state, Real dt) {
}

void linoz(State &state, Real dt) {
}

void step(State &state, Real dt) {
  size_t n_lev = state.atmosphere.temperature.size();
 
  // For guidance here, see MAMMicrophysics::run_impl in EAMxx.

  // TODO: add chlorine loading for linoz
  // TODO: add elevated emissions
  // TODO: add insolation based on orbital parameters/date and location (or just directly)
 
  // MAMMicrophysics::run_microphysics_kernels
  external_forcing(state, dt);
  set_invariants(state, dt);
  compute_o3_column_density(state, dt);
  if constexpr (mam4::nlev > 1) {
    photo_table(state, dt);
    sethet(state, dt);
  }
  drydep_xactive(state, dt);
  gas_phase_chemistry(state, dt);
  setsox(state, dt);
  modal_aero_amicphys_intr(state, dt);
  vmr2mmr_cw(state, dt);
  linoz(state, dt);

}

void update_diagnostics(const State &state, Real dt, Diagnostics &diagnostics) {
  diagnostics = {};
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
  State state = {};
  Diagnostics diagnostics = {};

  initialize(config, state);

  for (int i = 0; i < config.time.nstep; ++i) {
    step(state, config.time.dt);
    update_diagnostics(state, config.time.dt, diagnostics);
  }

  Result result = {};
  finalize(state, result);
  return result;
}

}
