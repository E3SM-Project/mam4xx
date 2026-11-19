// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "mamboxx.hpp"
#include <mam4xx/mam4_math.hpp>

namespace mamboxx {

namespace {

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

TimeConfig read_time(PyObject *time) {
  TimeConfig cfg = {};
  cfg.dt = get_real_field("time", time, "dt");
  cfg.nstep = get_int_field("time", time, "nstep");
  EKAT_REQUIRE(cfg.dt > 0.0, "Non-positive time.dt: " << cfg.dt);
  EKAT_REQUIRE(cfg.nstep > 0, "Non-positive time.nstep: " << cfg.nstep);
  return cfg;
}

mam4::microphysics::Config read_microphysics_config(PyObject *microphys) {
  mam4::microphysics::Config config = {};
  if (PyDict_ContainsString(microphys, "gas_chemistry"))
    config.gas_chemistry = get_bool_field("microphysics", microphys, "gas_chemistry");
  if (PyDict_ContainsString(microphys, "gas_aerosol_exchange"))
    config.gas_aerosol_exchange = get_bool_field("microphysics", microphys, "gas_aerosol_exchange");
  if (PyDict_ContainsString(microphys, "rename"))
    config.rename = get_bool_field("microphysics", microphys, "rename");
  if (PyDict_ContainsString(microphys, "nucleation"))
    config.nucleation = get_bool_field("microphysics", microphys, "nucleation");
  if (PyDict_ContainsString(microphys, "coagulation"))
    config.nucleation = get_bool_field("microphysics", microphys, "coagulation");
  return config;
}

ControlConfig read_control(PyObject *control) {
  ControlConfig cfg = {};
  if (PyDict_ContainsString(control, "microphysics"))
    cfg.microphysics = read_microphysics_config(PyDict_GetItemString(control, "microphysics"));
  return cfg;
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

void compute_mass_mixing_ratios_from_mass_fractions(const AtmosphereConfig &atmosphere,
                                                    const std::map<std::string, std::vector<Real>> &mass_fractions,
                                                    AerosolModeConfig &mode) {

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

  Real gmd = modal_geometric_mean_diameters.find(mode.name)->second;
  Real sigma = modal_geometric_mean_std_dev.find(mode.name)->second;
  size_t nlev = mode.number_mixing_ratio.size();

  for (auto iter = aero_ids_for_names.begin(); iter != aero_ids_for_names.end(); ++iter) {
    std::string aero_name = iter->first;
    mode.mass_mixing_ratios[aero_name] = std::vector<Real>(nlev);
  }

  // this isn't performant, but we don't care in a box/column model
  for (size_t k = 0; k < nlev; ++k) {
    Real total_aero_volume = mode.number_mixing_ratio[k] * 
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
      mode.mass_mixing_ratios[aero_name][k] = aero_mass_fractions[k] * total_aero_mass_mixing_ratio;
    }
  }
}

AerosolModeConfig read_aerosol_mode(const AtmosphereConfig &atmosphere, PyObject *mode,
                                    const char *mode_name) {
  AerosolModeConfig cfg = {};
  cfg.name = mode_name;

  EKAT_REQUIRE(PyDict_Check(mode), "aerosols.modes." << mode_name << " must be a dict");
  EKAT_REQUIRE(PyDict_ContainsString(mode, "number_concentration"),
               "aerosols.modes." << mode_name << ".number_concentration not specified");
  std::vector<Real> numc = read_box_or_column_data("aerosols.modes", mode, "number_concentration");
  size_t nlev = numc.size();

  // calculate number mixing ratio from number concentration
  cfg.number_mixing_ratio.resize(nlev);
  for (size_t k = 0; k < nlev; ++k) {
    EKAT_REQUIRE(numc[k] >= 0.0,
        "Negative aerosol number concentration at level " << k << " in mode " << cfg.name);
    // # of moles air / volume = pressure / (R x temperature) by the ideal gas law
    Real air_molar_concentration = atmosphere.pressure[k] / (mam4::Constants::r_gas_dry_air * atmosphere.temperature[k]);
    cfg.number_mixing_ratio[k] = numc[k] / air_molar_concentration / mam4::Constants::molec_weight_dry_air;
  }

  // read aerosol species mass fractions
  std::map<std::string, std::vector<Real>> mass_fractions;
  std::vector total_mass_fraction(nlev, 0.0);
  EKAT_REQUIRE(PyDict_ContainsString(mode, "mass_fractions"),
               "aerosols." << mode_name << ".mass_fractions not specified");
  PyObject *mf = PyDict_GetItemString(mode, "mass_fractions");
  Py_ssize_t pos = 0;
  PyObject *key, *mf_value;
  while (PyDict_Next(mf, &pos, &key, &mf_value)) {
    EKAT_REQUIRE(PyUnicode_Check(key), "At least one aerosol mode name is not a string");
    std::string species_name = PyUnicode_AsUTF8(key);
    auto section_name = std::string("aerosols.") + mode_name + "." + species_name;
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

  compute_mass_mixing_ratios_from_mass_fractions(atmosphere, mass_fractions, cfg);
  
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
    cfg.modes[mode_name] = read_aerosol_mode(atmosphere, mode, mode_name);
  }
  size_t n_levels = 0;
  for (auto iter = cfg.modes.begin(); iter != cfg.modes.end(); ++iter) {
    size_t n = iter->second.number_mixing_ratio.size();
    if (n_levels == 0) {
      n_levels = n;
    } else {
      EKAT_REQUIRE(n == n_levels,
          "found different numbers of vertical levels in aerosol modes: " << n_levels << ", " << n);
    }
  }
  return cfg;
}

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
    "droplet_activation",
    "output",
    "verification",
    nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "$sO!O!O!O!O!O!O!O!O!", keyword_list, &name,
                                   &PyDict_Type, &parameter_sweep, &PyDict_Type, &time,
                                   &PyDict_Type, &control, &PyDict_Type, &atmosphere,
                                   &PyDict_Type, &aerosols, &PyDict_Type, &gases,
                                   &PyDict_Type, &droplet_activation, &PyDict_Type, &output,
                                   &PyDict_Type, &verification))
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

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *column(PyObject *self, PyObject *args, PyObject *keywords) {
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
    "droplet_activation",
    "output",
    "verification",
    nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "$sO!O!O!O!O!O!O!O!O!", keyword_list, &name,
                                   &PyDict_Type, &parameter_sweep, &PyDict_Type, &time,
                                   &PyDict_Type, &control, &PyDict_Type, &atmosphere,
                                   &PyDict_Type, &aerosols, &PyDict_Type, &gases,
                                   &PyDict_Type, &droplet_activation, &PyDict_Type, &output,
                                   &PyDict_Type, &verification))
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

#if MAM4XX_NUM_VERTICAL_LEVELS == 1
  // box model function
  PyObject *box_function = PyCFunction_New(&box_def, (PyObject*)&config);
  PyDict_SetItemString(globals, "box", box_function);
#else
  // column model function
  PyObject *column_function = PyCFunction_New(&column_def, (PyObject*)&config);
  PyDict_SetItemString(globals, "column", column_function);
#endif

  return globals;
}

} // anonymous namespace

Config read_config(const std::string &filename) {
  Config config = {};

  EKAT_REQUIRE(not Py_IsInitialized(), "Python interpreter already initialized for some reason!");
  auto *globals = register_python_globals(config);

  FILE *f = std::fopen(filename.c_str(), "r");
  PyRun_FileEx(f, filename.c_str(), Py_file_input, globals, nullptr, 1);

  size_t num_modes = config.aerosols.modes.size();
  EKAT_REQUIRE(num_modes > 0, "no aerosol modes specified");

  // verify that # of vertical levels is the same everywhere
  size_t n_atm  = config.atmosphere.temperature.size();
  size_t n_aero = config.aerosols.modes.begin()->second.number_mixing_ratio.size();
  size_t n_gas = config.aerosols.modes.size();
  EKAT_REQUIRE(n_atm == n_aero == n_gas,
      "number of vertical levels differs for atmosphere (" << n_atm << "), aerosols (" << n_aero <<
      "), and gases (" << n_gas);

  return config;
}

}
