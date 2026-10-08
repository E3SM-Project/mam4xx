// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "ekat_assert.hpp"
#include "ekat_parameter_list.hpp"
#include "mamboxx.hpp"

#include <ekat_yaml.hpp>

namespace mamboxx {

namespace {

TimeConfig read_time(const ekat::ParameterList &params) {
  TimeConfig cfg = {};
  cfg.dt = static_cast<Real>(params.get<double>("dt"));
  cfg.nstep = params.get<int>("nstep");
  EKAT_REQUIRE(cfg.dt > 0.0, "Non-positive time.dt: " << cfg.dt);
  EKAT_REQUIRE(cfg.nstep > 0, "Non-positive time.nstep: " << cfg.nstep);
  return cfg;
}

ControlConfig read_control(const ekat::ParameterList &params) {
  ControlConfig cfg = {};
  if (params.isParameter("gas_chemistry"))
    cfg.gas_chemistry = params.get<bool>("gas_chemistry");
  if (params.isParameter("gas_aerosol_exchange"))
    cfg.gas_aerosol_exchange = params.get<bool>("gas_aerosol_exchange");
  if (params.isParameter("rename"))
    cfg.nucleation = params.get<bool>("rename");
  if (params.isParameter("nucleation"))
    cfg.nucleation = params.get<bool>("nucleation");
  if (params.isParameter("coagulation"))
    cfg.coagulation = params.get<bool>("coagulation");
  return cfg;
}

std::vector<Real> read_box_or_column(const ekat::ParameterList &params, const std::string &name) {
  EKAT_REQUIRE(params.isType<double>(name) or params.isType<std::vector<double>>(name),
    params.name() << "." << name << " is neither a scalar nor a column");
  std::vector<Real> values;
  if (params.isType<double>(name)) { // box, not column
    values = {static_cast<Real>(params.get<double>(name))};
  } else {
    auto doubles = params.get<std::vector<double>>(name);
    values.resize(doubles.size());
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<Real>(doubles[i]);
    }
  }
  return values;
}

AtmosphereConfig read_atmosphere(const ekat::ParameterList &params) {
  AtmosphereConfig cfg = {};
  EKAT_REQUIRE(params.isParameter("temperature"), "No atmosphere.temperature specified");
  EKAT_REQUIRE(params.isParameter("pressure"), "No atmosphere.pressure specified");
  EKAT_REQUIRE(params.isParameter("relative_humidity"), "No atmosphere.relative_humidity specified");

  cfg.temperature = read_box_or_column(params, "temperature");
  cfg.pressure = read_box_or_column(params, "pressure");
  cfg.relative_humidity = read_box_or_column(params, "relative_humidity");

  return cfg;
}

AerosolModeConfig read_aerosol_mode(const ekat::ParameterList &params) {
  AerosolModeConfig cfg = {};
  EKAT_REQUIRE(params.isParameter("numc"), "no numc specified for aerosol mode " << params.name());
  cfg.name = params.name();
  cfg.numc = read_box_or_column(params, "numc");
  for (size_t i = 0; i < cfg.numc.size(); ++i) {
    EKAT_REQUIRE(cfg.numc[i] >= 0.0,
        "Negative aerosol number concentration at level " << i << " in mode " << cfg.name);
  }
  auto species_names = params.param_names();
  std::vector total_mass_fraction(cfg.numc.size(), 0.0);
  for (const std::string &species: species_names) {
    if (species != "numc") { // numc's not a species, silly
      auto mass_fraction = read_box_or_column(params, species);
      EKAT_REQUIRE(mass_fraction.size() == cfg.numc.size(),
          "Aerosol mass fraction for species " << species << " has different number of values (" <<
          mass_fraction.size() << ") than numc (" << cfg.numc.size() << " in aerosol mode " << cfg.name);
      for (size_t i = 0; i < mass_fraction.size(); ++i) {
        EKAT_REQUIRE(mass_fraction[i] >= 0.0,
            "Negative aerosol mass fraction for species " << species << " at level " << i << " in mode " << cfg.name);
        total_mass_fraction[i] += mass_fraction[i];
      }
      cfg.mass_fractions[species] = mass_fraction;
    }
  }
  for (size_t i = 0; i < total_mass_fraction.size(); ++i) {
    EKAT_REQUIRE(total_mass_fraction[i] <= 1.0, 
        "Total aerosol mass fraction in mode " << cfg.name << " exceeds 1");
  }
  
  return cfg;
}

AerosolState read_aerosols(const ekat::ParameterList &params) {
  AerosolState cfg = {};
  EKAT_REQUIRE(params.isSublist("modes"), "No aerosol.modes specified");
  auto modes = params.sublist("modes");
  auto mode_names = modes.sublist_names(); // NOTE: sublists interpreted as modes
  for (size_t i = 0; i < mode_names.size(); ++i) {
    auto mode = modes.sublist(mode_names[i]);
    cfg.modes[mode_names[i]] = read_aerosol_mode(mode);
  }
  return cfg;
}

GasState read_gases(const ekat::ParameterList &params) {
  GasState cfg = {};
  auto gas_names = params.param_names();
  for (const std::string &gas_name: gas_names) {
    auto mass_mixing_ratios = read_box_or_column(params, gas_name);
    for (size_t i = 0; i < mass_mixing_ratios.size(); ++i) {
      EKAT_REQUIRE(mass_mixing_ratios[i] >= 0.0,
          "Negative mass mixing ratio at level " << i << " for gas " << gas_name);
    }
    cfg.mass_mixing_ratios[gas_name] = mass_mixing_ratios;
  }
  return cfg;
}

} // anonymous namespace

Config read_config(const std::string &filename) {

  auto params = ekat::parse_yaml_file(filename);

  EKAT_REQUIRE(params.isSublist("time"), "No time section found in " << filename);
  EKAT_REQUIRE(params.isSublist("control"), "No control section found in " << filename);
  EKAT_REQUIRE(params.isSublist("atmosphere"), "No atmosphere section found in " << filename);
  EKAT_REQUIRE(params.isSublist("aerosols"), "No aerosols section found in " << filename);
  EKAT_REQUIRE(params.isSublist("gases"), "No gases section found in " << filename);

  auto time = params.sublist("time");
  auto control = params.sublist("control");
  auto atmosphere = params.sublist("atmosphere");
  auto aerosols = params.sublist("aerosols");
  auto gases = params.sublist("gases");

  Config cfg = {};
  cfg.time = read_time(time);
  cfg.control = read_control(control);
  cfg.atmosphere = read_atmosphere(atmosphere);
  cfg.aerosols = read_aerosols(aerosols);
  cfg.gases = read_gases(aerosols);

  return cfg;
}

}
