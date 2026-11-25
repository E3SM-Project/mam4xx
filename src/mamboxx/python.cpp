// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "mamboxx.hpp"
#include "mamboxx/aerosol_cloud_interaction.hpp"
#include "mamboxx/microphysics.hpp"
#include <mam4xx/mam4_math.hpp>

#include <array>

namespace mamboxx {

namespace {

#if Python3_VERSION_MINOR < 13

int PyDict_ContainsString(PyObject *dict, const char *field) {
  PyObject *py_field = PyUnicode_FromString(field);
  int result = PyDict_Contains(dict, py_field);
  Py_DECREF(py_field);
  return result;
}

int PyLong_AsInt(PyObject *obj) {
  return (int)PyLong_AsLong(obj);
}

#endif

//---------------------------
// Reading data and metadata
//---------------------------

int get_int_field(const char *section, PyObject *dict, const char *field) {
  EKAT_REQUIRE(PyDict_ContainsString(dict, field), "section " << section << " is missing " << field);
  PyObject *value = PyDict_GetItemString(dict, field);
  EKAT_REQUIRE(PyLong_Check(value), "provided " << field << " in section " << section << " is not an integer");
  return PyLong_AsInt(value);
}

Real get_real_field(const char *section, PyObject *dict, const char *field) {
  EKAT_REQUIRE(PyDict_ContainsString(dict, field), "section " << section << " is missing " << field);
  PyObject *value = PyDict_GetItemString(dict, field);
  EKAT_REQUIRE(PyFloat_Check(value), "provided " << field << " in section " << section << " is not a float");
  return static_cast<Real>(PyFloat_AsDouble(value));
}

bool get_bool_field(const char *section, PyObject *dict, const char *field) {
  EKAT_REQUIRE(PyDict_ContainsString(dict, field), "section " << section << " is missing " << field);
  PyObject *value = PyDict_GetItemString(dict, field);
  EKAT_REQUIRE(PyBool_Check(value), "provided " << field << " in section " << section << " is not a boolean");
  return value == Py_True;
}

std::string get_string_field(const char *section, PyObject *dict, const char *field) {
  EKAT_REQUIRE(PyDict_ContainsString(dict, field), "section " << section << " is missing " << field);
  PyObject *value = PyDict_GetItemString(dict, field);
  EKAT_REQUIRE(PyUnicode_Check(value), "provided " << field << " in section " << section << " is not a string");
  return std::string(PyUnicode_AsUTF8(value));
}

template <typename T, unsigned long N>
T select_enum_from_string(std::string name, std::string value,
                          std::array<std::string, N>    options) {
  for (unsigned long i = 0; i < N; ++i) {
    if (value == options[i])
      return static_cast<T>(i);
  }
  EKAT_REQUIRE(false, "Invalid enum option for " << name << ": " << value);
  return static_cast<T>(-1);
}

std::vector<Real> read_box_or_column_data(const char * section, PyObject *value, const std::string &name) {
  EKAT_REQUIRE(PyFloat_Check(value) or PySequence_Check(value),
    section << "." << name << " is neither a scalar nor a column");
  Py_ssize_t size = PyFloat_Check(value) ? 1 : PySequence_Size(value);
  std::vector<Real> values(size);
  if (PyFloat_Check(value)) { // scalar
    values[0] = static_cast<Real>(PyFloat_AsDouble(value));
  } else { // sequence
    PyObject *seq = PySequence_Fast(value, "Not a sequence!");
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *vi = PySequence_Fast_GET_ITEM(seq, i);
      EKAT_REQUIRE(PyFloat_Check(vi), "Item " << i << " in " << section << "." << name << " is not a float!");
      values[i] = PyFloat_AsDouble(vi);
    }
    Py_DECREF(seq);
  }
  return values;
}

//----------------------------
// Timestepping configuration
//----------------------------

TimeConfig read_time(PyObject *time) {
  TimeConfig cfg = {};
  cfg.dt = get_real_field("time", time, "dt");
  cfg.nstep = get_int_field("time", time, "nstep");
  EKAT_REQUIRE(cfg.dt > 0.0, "Non-positive time.dt: " << cfg.dt);
  EKAT_REQUIRE(cfg.nstep > 0, "Non-positive time.nstep: " << cfg.nstep);
  return cfg;
}

//------------------------------------
// Aerosol microphysics configuration
//------------------------------------

mam4::microphysics::GasAerosolExchangeConfig
read_gas_aerosol_exchange_config(PyObject *gas_aerosol_exchange) {
  return mam4::microphysics::GasAerosolExchangeConfig{
    .enable = get_bool_field("microphysics.gas_aerosol_exchange", gas_aerosol_exchange, "enable"),
  };
}

mam4::microphysics::CoagulationConfig read_coagulation_config(PyObject *coagulation) {
  return mam4::microphysics::CoagulationConfig{
    .enable = get_bool_field("microphysics.coagulation", coagulation, "enable"),
  };
}

mam4::microphysics::GasChemistryConfig read_gas_chemistry_config(PyObject *gas_chemistry) {
  return mam4::microphysics::GasChemistryConfig{
    .enable = get_bool_field("microphysics.gas_chemistry", gas_chemistry, "enable"),
  };
}

mam4::microphysics::RenameConfig read_rename_config(PyObject *rename) {
  return mam4::microphysics::RenameConfig{
    .enable = get_bool_field("microphysics.rename", rename, "enable"),
  };
}

mam4::microphysics::NucleationConfig read_nucleation_config(PyObject *nucleation) {
  return mam4::microphysics::NucleationConfig{
    .enable = get_bool_field("microphysics.nucleation", nucleation, "enable"),
  };
}

mam4::microphysics::Config read_microphysics_config(PyObject *microphys) {
  mam4::microphysics::Config config = {};
  if (PyDict_ContainsString(microphys, "coagulation")) {
    PyObject *coagulation = PyDict_GetItemString(microphys, "coagulation");
    config.coagulation = read_coagulation_config(coagulation);
  }
  if (PyDict_ContainsString(microphys, "gas_aerosol_exchange")) {
    PyObject *gas_aerosol_exchange = PyDict_GetItemString(microphys, "gas_aerosol_exchange");
    config.gas_aerosol_exchange = read_gas_aerosol_exchange_config(gas_aerosol_exchange);
  }
  if (PyDict_ContainsString(microphys, "gas_chemistry")) {
    PyObject *gas_chemistry = PyDict_GetItemString(microphys, "gas_chemistry");
    config.gas_chemistry = read_gas_chemistry_config(gas_chemistry);
  }
  if (PyDict_ContainsString(microphys, "rename")) {
    PyObject *rename = PyDict_GetItemString(microphys, "rename");
    config.rename = read_rename_config(rename);
  }
  if (PyDict_ContainsString(microphys, "nucleation")) {
    PyObject *nucleation = PyDict_GetItemString(microphys, "nucleation");
    config.nucleation = read_nucleation_config(nucleation);
  }
  return config;
}

//-----------------------------------------
// Aerosol-Cloud interaction configuration
//-----------------------------------------

mam4::aerosol_cloud_interaction::DropletActivationConfig
read_droplet_activation_config(PyObject *droplet_activation) {
  using CriticalRelation = mam4::aerosol_cloud_interaction::CriticalRelation;
  static std::array<std::string, 1> critical_relations = {
    "dilute_kappa_kohler",
  };
  static std::array<CriticalRelation, 1> critical_relation_enums = {
      CriticalRelation::DiluteKappaKohler,
    };
  std::string critical_relation =
    get_string_field("aerosol_cloud_interaction.critical_relation",
                     droplet_activation, "critical_relation");
  return mam4::aerosol_cloud_interaction::DropletActivationConfig{
    .enable = get_bool_field("aerosol_cloud_interaction.droplet_activation",
                             droplet_activation, "enable"),
    .critical_relation = select_enum_from_string<CriticalRelation>("aerosol_cloud_interaction.critical_relation",
                                                 critical_relation, critical_relations),
  };
}

mam4::aerosol_cloud_interaction::IceNucleationConfig
read_ice_nucleation_config(PyObject *ice_nucleation) {
  return mam4::aerosol_cloud_interaction::IceNucleationConfig{
    .enable = get_bool_field("aerosol_cloud_interaction.ice_nucleation",
                             ice_nucleation, "enable"),
    .subgrid = get_real_field("aerosol_cloud_interaction.ice_nucleation",
                              ice_nucleation, "subgrid"),
    .so4_size_threshold = get_real_field("aerosol_cloud_interaction.ice_nucleation",
                                         ice_nucleation, "so4_size_threshold"),
  };
}

mam4::aerosol_cloud_interaction::Config read_aerosol_cloud_interaction_config(PyObject *aci) {
  mam4::aerosol_cloud_interaction::Config config = {};
  if (PyDict_ContainsString(aci, "droplet_activation")) {
    PyObject *droplet_activation = PyDict_GetItemString(aci, "droplet_activation");
    config.dropmixnuc.droplet_activation = read_droplet_activation_config(droplet_activation);
  }
  if (PyDict_ContainsString(aci, "ice_nucleation")) {
    PyObject *ice_nucleation = PyDict_GetItemString(aci, "ice_nucleation");
    config.dropmixnuc.ice_nucleation = read_ice_nucleation_config(ice_nucleation);
  }
  return config;
}

//-----------------------------------------------
// Omni-parameterization configuration (control)
//-----------------------------------------------

ControlConfig read_control(PyObject *control) {
  ControlConfig cfg = {};
  if (PyDict_ContainsString(control, "microphysics")) {
    PyObject *microphysics = PyDict_GetItemString(control, "microphysics");
    cfg.microphysics = read_microphysics_config(microphysics);
  }
  if (PyDict_ContainsString(control, "gas_aerosol_exchange")) {
    PyObject *aci = PyDict_GetItemString(control, "aerosol_cloud_interaction");
    cfg.aerosol_cloud_interaction = read_aerosol_cloud_interaction_config(aci);
  }
  return cfg;
}

//---------------------------------
// Atmospheric state configuration
//---------------------------------

AtmosphereConfig read_atmosphere(PyObject *atmosphere) {
  AtmosphereConfig cfg = {};
  EKAT_REQUIRE(PyDict_ContainsString(atmosphere, "temperature"),
               "No atmosphere.temperature specified");
  EKAT_REQUIRE(PyDict_ContainsString(atmosphere, "pressure"),
               "No atmosphere.pressure specified");
  EKAT_REQUIRE(PyDict_ContainsString(atmosphere, "relative_humidity"),
               "No atmosphere.relative_humidity specified");

  cfg.temperature = read_box_or_column_data("atmosphere", atmosphere, "temperature");
  cfg.pressure = read_box_or_column_data("atmosphere", atmosphere, "pressure");
  cfg.relative_humidity = read_box_or_column_data("atmosphere", atmosphere, "relative_humidity");

  EKAT_REQUIRE(cfg.temperature.size() == cfg.pressure.size() and
               cfg.temperature.size() == cfg.relative_humidity.size(),
               "size mismatch in atmosphere data: all arrays must be the same length");

  for (size_t i = 0; i < cfg.temperature.size(); ++i) {
    EKAT_REQUIRE(cfg.temperature[i] > 0.0,
        "non-positive atmosphere.temperature[" << i << "]: " << cfg.temperature[i]);
    EKAT_REQUIRE(cfg.pressure[i] > 0.0,
        "non-positive atmosphere.pressure[" << i << "]: " << cfg.pressure[i]);
    EKAT_REQUIRE(cfg.relative_humidity[i] >= 0.0 and cfg.relative_humidity[i] <= 1.0,
        "invalid atmosphere.relative_humidity[" << i << "]: " << cfg.relative_humidity[i]);
  }

  // compute the water vapor mixing ratio

  return cfg;
}

//-----------------------------
// Aerosol state configuration
//-----------------------------

void compute_mass_mixing_ratios_from_mass_fractions(const AtmosphereConfig &atmosphere,
                                                    const std::map<std::string, std::vector<Real>> &mass_fractions,
                                                    const std::string &mode_name,
                                                    AerosolMixingRatiosConfig &mixing_ratios) {

  // FIXME: need to check these quantities
  const std::map<std::string, Real> modal_geometric_mean_diameters = {
    {"accumulation", mam4::mam4_accum_nom_diameter_m},
    {"aitken", mam4::mam4_aitken_nom_diameter_m},
    {"coarse", mam4::mam4_coarse_nom_diameter_m},
    {"primary_carbon", mam4::mam4_primary_carbon_nom_diameter_m},
  };
  const std::map<std::string, Real> modal_geometric_mean_std_dev = {
    {"accumulation", mam4::mam4_accum_mead_std_dev},
    {"aitken", mam4::mam4_aitken_mead_std_dev},
    {"coarse", mam4::mam4_coarse_mead_std_dev},
    {"primary_carbon", mam4::mam4_primary_carbon_mead_std_dev},
  };
  const std::map<std::string, Real> aero_mass_density = {
    {"bc", mam4::defaults::mam4_density_bc},
    {"dst", mam4::defaults::mam4_density_dst},
    {"mom", mam4::defaults::mam4_density_mom},
    {"ncl", mam4::defaults::mam4_density_nacl},
    {"pom", mam4::defaults::mam4_density_pom},
    {"soa", mam4::defaults::mam4_density_soa},
    {"so4", mam4::defaults::mam4_density_so4},
  };

  Real gmd = modal_geometric_mean_diameters.find(mode_name)->second;
  Real sigma = modal_geometric_mean_std_dev.find(mode_name)->second;
  size_t nlev = mixing_ratios.number_mixing_ratio.size();

  for (auto iter = aero_ids_for_names.begin(); iter != aero_ids_for_names.end(); ++iter) {
    std::string aero_name = iter->first;
    mixing_ratios.mass_mixing_ratios[aero_name] = std::vector<Real>(nlev);
    mixing_ratios.mass_mixing_ratios[aero_name] = std::vector<Real>(nlev);
  }


  // this isn't performant, but we don't care in a box/column model
  for (size_t k = 0; k < nlev; ++k) {
    Real total_aero_volume = mixing_ratios.number_mixing_ratio[k] * 
                             mam4::cube(gmd) * (M_PI/6.0) * mam4::exp(4.5*sigma*sigma);
    Real inv_total_aero_density = 0.0;
    for (auto iter = aero_ids_for_names.begin(); iter != aero_ids_for_names.end(); ++iter) {
      std::string  aero_name = iter->first;
      mam4::AeroId aero_id = iter->second;
      auto mf_iter = mass_fractions.find(aero_name);
      EKAT_ASSERT(mf_iter != mass_fractions.end());
      auto rho_iter = aero_mass_density.find(aero_name);
      EKAT_ASSERT(rho_iter != aero_mass_density.end());
      const std::vector<Real> &aero_mass_fractions = mf_iter->second;
      Real aero_mass_density = rho_iter->second;
      inv_total_aero_density += aero_mass_fractions[k] / aero_mass_density;
    }
    Real total_aero_mass_mixing_ratio = total_aero_volume / inv_total_aero_density;
    for (auto iter = aero_ids_for_names.begin(); iter != aero_ids_for_names.end(); ++iter) {
      std::string  aero_name = iter->first;
      mam4::AeroId aero_id = iter->second;
      const std::vector<Real> &aero_mass_fractions = mass_fractions.find(aero_name)->second;
      mixing_ratios.mass_mixing_ratios[aero_name][k] =
        aero_mass_fractions[k] * total_aero_mass_mixing_ratio;
    }
  }
}

AerosolMixingRatiosConfig read_aerosol_mixing_ratios(const AtmosphereConfig &atmosphere,
                                                     PyObject *mixing_ratios, const char *which,
                                                     const char *mode_name) {
  AerosolMixingRatiosConfig cfg = {};
  EKAT_REQUIRE(PyDict_Check(mixing_ratios), std::string(which) << "mixing ratios must be a dict");
  EKAT_REQUIRE(PyDict_ContainsString(mixing_ratios, "number_concentration"),
               "aerosols.modes." << mode_name << ".number_concentration not specified");
  auto ratios_label = std::string("aerosols.modes.") + std::string(which);
  std::vector<Real> numc = read_box_or_column_data(ratios_label.c_str(), mixing_ratios, "number_concentration");
  size_t nlev = numc.size();

  // calculate number mixing ratio from number concentration
  cfg.number_mixing_ratio.resize(nlev);
  for (size_t k = 0; k < nlev; ++k) {
    EKAT_REQUIRE(numc[k] >= 0.0,
        "Negative " << which << " aerosol number concentration at level " << k <<
        " in mode " << mode_name);
    // # of moles air / volume = pressure / (R x temperature) by the ideal gas law
    Real air_molar_concentration = atmosphere.pressure[k] / (mam4::Constants::r_gas_dry_air * atmosphere.temperature[k]);
    cfg.number_mixing_ratio[k] = numc[k] / air_molar_concentration / mam4::Constants::molec_weight_dry_air;
  }

  // read aerosol species mass fractions
  std::map<std::string, std::vector<Real>> mass_fractions;
  std::vector total_mass_fraction(nlev, 0.0);
  EKAT_REQUIRE(PyDict_ContainsString(mixing_ratios, "mass_fractions"),
               ratios_label << ".mass_fractions not specified");
  PyObject *mf = PyDict_GetItemString(mixing_ratios, "mass_fractions");
  Py_ssize_t pos = 0;
  PyObject *key, *mf_value;
  while (PyDict_Next(mf, &pos, &key, &mf_value)) {
    EKAT_REQUIRE(PyUnicode_Check(key), "At least one aerosol mode name is not a string");
    std::string species_name = PyUnicode_AsUTF8(key);
    auto section_name = std::string("aerosols.") + mode_name + "." + which + "." + species_name;
    auto mass_fraction = read_box_or_column_data(section_name.c_str(), mf_value, species_name);
    for (size_t k = 0; k < mass_fraction.size(); ++k) {
      EKAT_REQUIRE(mass_fraction[k] >= 0.0,
        "aerosols." << mode_name << ".mass_fractions." << species_name << "[" << k << "] is negative");
      total_mass_fraction[k] += mass_fraction[k];
    }
  }
  for (size_t k = 0; k < total_mass_fraction.size(); ++k) {
    EKAT_REQUIRE(total_mass_fraction[k] <= 1.0, 
        "Total aerosol mass fraction in mode " << mode_name << " exceeds 1");
  }

  compute_mass_mixing_ratios_from_mass_fractions(atmosphere, mass_fractions, mode_name, cfg);
  
  return cfg;
}

AerosolModeConfig read_aerosol_mode(const AtmosphereConfig &atmosphere,
                                    PyObject *mode, const char *mode_name) {
  EKAT_REQUIRE(PyDict_Check(mode), "aerosols.modes." << mode_name << " must be a dict");

  AerosolModeConfig cfg = {};
  cfg.name = mode_name;

  PyObject *interstitial = PyDict_GetItemString(mode, "interstitial");
  EKAT_REQUIRE(interstitial != nullptr, "interstitial composition is not specified");
  cfg.interstitial = read_aerosol_mixing_ratios(atmosphere, interstitial, "interstitial", mode_name);

  PyObject *cloudborne = PyDict_GetItemString(mode, "cloudborne");
  EKAT_REQUIRE(cloudborne != nullptr, "cloudborne composition is not specified");
  cfg.cloudborne = read_aerosol_mixing_ratios(atmosphere, cloudborne, "cloudborne", mode_name);
  return cfg;
}

AerosolConfig read_aerosols(const AtmosphereConfig &atmosphere, PyObject *aerosols) {
  AerosolConfig cfg = {};
  EKAT_REQUIRE(PyDict_ContainsString(aerosols, "modes"), "No aerosol.modes specified");
  PyObject *modes = PyDict_GetItemString(aerosols, "modes");
  EKAT_REQUIRE(PyDict_Check(modes), "aerosol.modes must be a dict");
  Py_ssize_t pos = 0;
  PyObject *key, *mode;
  while (PyDict_Next(modes, &pos, &key, &mode)) {
    EKAT_REQUIRE(PyUnicode_Check(key), "At least one aerosol mode name is not a string");
    const char *mode_name = PyUnicode_AsUTF8(key);
    printf("Reading aerosol mode %s\n", mode_name);
    cfg.modes[mode_name] = read_aerosol_mode(atmosphere, mode, mode_name);
  }
  size_t n_levels = 0;
  for (auto iter = cfg.modes.begin(); iter != cfg.modes.end(); ++iter) {
    size_t n = iter->second.interstitial.number_mixing_ratio.size();
    if (n_levels == 0) {
      n_levels = n;
    } else {
      EKAT_REQUIRE(n == n_levels,
          "found different numbers of vertical levels in aerosol modes: " << n_levels << ", " << n);
    }
  }
  return cfg;
}

//-------------------------
// Gas state configuration
//-------------------------

GasConfig read_gases(PyObject *gases) {
  GasConfig cfg = {};
  EKAT_REQUIRE(PyDict_Check(gases), "gases must be a dict");
  Py_ssize_t pos = 0;
  PyObject *key, *mode;
  while (PyDict_Next(gases, &pos, &key, &mode)) {
    EKAT_REQUIRE(PyUnicode_Check(key), "At least one gas name is not a string");
    const char *gas_name = PyUnicode_AsUTF8(key);
    auto mass_mixing_ratios = read_box_or_column_data("gases", gases, gas_name);
    for (size_t k = 0; k < mass_mixing_ratios.size(); ++k) {
      EKAT_REQUIRE(mass_mixing_ratios[k] >= 0.0,
          "Negative mass mixing ratio at level " << k << " for gas " << gas_name);
    }
    cfg.mass_mixing_ratios[gas_name] = mass_mixing_ratios;
  }
  size_t n_levels = 0;
  for (auto iter = cfg.mass_mixing_ratios.begin(); iter != cfg.mass_mixing_ratios.end(); ++iter) {
    size_t n = iter->second.size();
    if (n_levels == 0) {
      n_levels = n;
    } else {
      EKAT_REQUIRE(n == n_levels,
          "found different numbers of vertical levels in gases: " << n_levels << ", " << n);
    }
  }
  return cfg;
}

//----------------------------
// Verification configuration
//----------------------------

std::vector<PyObject *> read_checks(PyObject *checks) {
  std::vector<PyObject *> check_functions;
  EKAT_REQUIRE(PySequence_Check(checks), "checks must be a list of functions that accept a state");
  Py_ssize_t num_checks = PySequence_Size(checks);
  for (Py_ssize_t i = 0; i < num_checks; ++i) {
    PyObject *check_i = PySequence_Fast_GET_ITEM(checks, i);
    EKAT_REQUIRE(PyCallable_Check(check_i), "Item " << i << " in verification_checks is not a callable function");
    check_functions.push_back(check_i);
  }
  return check_functions;
}

VerificationConfig read_verification(PyObject *verification) {
  VerificationConfig cfg = {};
  EKAT_REQUIRE(PyDict_Check(verification), "verification must be a dict");

  if (PyDict_ContainsString(verification, "checks")) {
    PyObject *checks = PyDict_GetItemString(verification, "checks");
    cfg.checks = read_checks(checks);
  }
  return cfg;
}

//---------------------
// Python entry points
//---------------------

PyObject *box(PyObject *self, PyObject *args, PyObject *keywords) {
  Config *config = (Config*)self;

  char     *name;
  PyObject *parameter_sweep, *time, *control, *atmosphere, *aerosols, *gases,
           *droplet_activation, *output, *verification;

  static const char *keyword_list[] = {
    "name",
    "parameter_sweep",
    "time",
    "control",
    "atmosphere",
    "aerosols",
    "gases",
    "output",
    "verification",
    nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "$sO!O!O!O!O!O!O!O!",
#if Python3_VERSION_MINOR < 13
        (char**)keyword_list,
#else
        keyword_list,
#endif
        &name, &PyDict_Type, &parameter_sweep, &PyDict_Type, &time, &PyDict_Type, &control,
        &PyDict_Type, &atmosphere, &PyDict_Type, &aerosols, &PyDict_Type, &gases,
        &PyDict_Type, &output, &PyDict_Type, &verification))
    return nullptr;

  // FIXME
  EKAT_REQUIRE(time != nullptr, "No time section found in input");
  EKAT_REQUIRE(control != nullptr, "No control section found in input");
  EKAT_REQUIRE(atmosphere != nullptr, "No atmosphere section found in input");
  EKAT_REQUIRE(aerosols != nullptr, "No aerosols section found in input");
  EKAT_REQUIRE(gases != nullptr, "No gases section found in input");

  config->name = name;
  config->time = read_time(time);
  config->control = read_control(control);
  config->atmosphere = read_atmosphere(atmosphere);
  config->aerosols = read_aerosols(config->atmosphere, aerosols);
  config->gases = read_gases(aerosols);
  config->verification = read_verification(verification);

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *column(PyObject *self, PyObject *args, PyObject *keywords) {
  Config *config = (Config*)self;

  char     *name;
  PyObject *parameter_sweep, *time, *control, *atmosphere, *aerosols, *gases,
           *output, *verification;

  static const char *keyword_list[] = {
    "name",
    "parameter_sweep",
    "time",
    "control",
    "atmosphere",
    "aerosols",
    "gases",
    "output",
    "verification",
    nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "$sO!O!O!O!O!O!O!O!",
#if Python3_VERSION_MINOR < 13
        (char**)keyword_list,
#else
        keyword_list,
#endif
        &name, &PyDict_Type, &parameter_sweep, &PyDict_Type, &time,
        &PyDict_Type, &control, &PyDict_Type, &atmosphere,
        &PyDict_Type, &aerosols, &PyDict_Type, &gases,
        &PyDict_Type, &output, &PyDict_Type, &verification)) {
    return nullptr;
  }

  // FIXME
  EKAT_REQUIRE(time != nullptr, "No time section found in input");
  EKAT_REQUIRE(control != nullptr, "No control section found in input");
  EKAT_REQUIRE(atmosphere != nullptr, "No atmosphere section found in input");
  EKAT_REQUIRE(aerosols != nullptr, "No aerosols section found in input");
  EKAT_REQUIRE(gases != nullptr, "No gases section found in input");

  config->name = name;
  config->time = read_time(time);
  config->control = read_control(control);
  config->atmosphere = read_atmosphere(atmosphere);
  config->aerosols = read_aerosols(config->atmosphere, aerosols);
  config->gases = read_gases(aerosols);

  Py_INCREF(Py_None);
  return Py_None;
}

PyMethodDef box_def = {
  "box",
  (PyCFunction)(void(*)(void))box,
  METH_VARARGS | METH_KEYWORDS,
  """box(...) - run a box model simulation",  // docstring (TODO: should be longer)
};

PyMethodDef column_def = {
  "column",
  (PyCFunction)(void(*)(void))box,
  METH_VARARGS | METH_KEYWORDS,
  """column(...) - run a single-column simulation",  // docstring (TODO: should be longer)
};

// Registers Python functions and objects as global objects, returning a dictionary of these
// globals.
PyObject* register_python_globals(Config &config) {
  Py_Initialize();
  std::atexit(Py_Finalize);

  PyObject *globals = PyDict_New();

  // convenient physical constants
  PyObject *constants = PyDict_New();
  PyDict_SetItemString(constants, "gas_constant", PyFloat_FromDouble(8.314563841815325));
  PyDict_SetItemString(globals, "constants", constants);

  if (mam4::nlev == 1) {
    // box model function
    PyObject *box_function = PyCFunction_New(&box_def, (PyObject*)&config);
    PyDict_SetItemString(globals, "box", box_function);
  } else {
    // column model function
    PyObject *column_function = PyCFunction_New(&column_def, (PyObject*)&config);
    PyDict_SetItemString(globals, "column", column_function);
  }

  return globals;
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

} // anonymous namespace

//----------------------------
// read_config implementation
//----------------------------

Config read_config(const std::string &filename) {
  Config config = {};

  EKAT_REQUIRE(not Py_IsInitialized(), "Python interpreter already initialized for some reason!");
  auto *globals = register_python_globals(config);

  printf("Reading %s\n", filename.c_str());
  FILE *f = std::fopen(filename.c_str(), "r");
  PyObject *locals = PyDict_New();
  PyObject *result = PyRun_FileEx(f, filename.c_str(), Py_file_input, globals, locals, 1);
  if (result == nullptr) {
    PyErr_Print();
    EKAT_REQUIRE(false, "An error occurred while reading input.");
  }

  size_t num_modes = config.aerosols.modes.size();
  EKAT_REQUIRE(num_modes > 0, "no aerosol modes specified");

  // verify that # of vertical levels is the same everywhere
  size_t n_atm    = config.atmosphere.temperature.size();
  size_t n_aero_i = config.aerosols.modes.begin()->second.interstitial.number_mixing_ratio.size();
  size_t n_aero_c = config.aerosols.modes.begin()->second.cloudborne.number_mixing_ratio.size();
  size_t n_gas    = config.aerosols.modes.size();
  EKAT_REQUIRE(n_atm == n_aero_i == n_aero_c == n_gas,
      "number of vertical levels differs for atmosphere (" << n_atm << "), aerosols (" <<
      n_aero_i << ", " << n_aero_c << "), and gases (" << n_gas);

  return config;
}

bool run_verification_checks(const Config &config, const State &state) {
  PyObject *py_state = create_python_state(state);
  for (auto check: config.verification.checks) {
    PyObject *result = PyObject_Call(check, py_state, NULL);
    EKAT_REQUIRE(PyBool_Check(result),
        "verification check functions must all return True or False");
    if (result != Py_True) {
      fprintf(stderr, "Verification check %s FAILED.",
          PyUnicode_AsUTF8(PyObject_GetAttrString(result, "__name__")));
      Py_DECREF(result);
      return false;
    }
    Py_DECREF(result);
  }
  fprintf(stderr, "All verification checks PASSED.");
  return true;
}

}
