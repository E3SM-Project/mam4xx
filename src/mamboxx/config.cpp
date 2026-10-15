// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "mamboxx.hpp"
#include <mam4xx/mam4_math.hpp>

#include <ekat_assert.hpp>
#include <ekat_parameter_list.hpp>
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

AerosolModeConfig read_aerosol_mode(const AtmosphereConfig &atmosphere, const ekat::ParameterList &params) {
  AerosolModeConfig cfg = {};
  cfg.name = params.name();

  EKAT_REQUIRE(params.isParameter("numc"), "no numc specified for aerosol mode " << params.name());
  std::vector<Real> numc = read_box_or_column(params, "numc");
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

  auto species_names = params.param_names();
  std::vector total_mass_fraction(nlev, 0.0);
  std::map<std::string, std::vector<Real>> mass_fractions;
  for (const std::string &species: species_names) {
    if (species != "numc") { // numc's not a species, silly
      auto mass_fraction = read_box_or_column(params, species);
      EKAT_REQUIRE(mass_fraction.size() == numc.size(),
          "Aerosol mass fraction for species " << species << " has different number of values (" <<
          mass_fraction.size() << ") than numc (" << nlev << " in aerosol mode " << cfg.name);
      for (size_t i = 0; i < mass_fraction.size(); ++i) {
        EKAT_REQUIRE(mass_fraction[i] >= 0.0,
            "Negative aerosol mass fraction for species " << species << " at level " << i << " in mode " << cfg.name);
        total_mass_fraction[i] += mass_fraction[i];
      }
      mass_fractions[species] = mass_fraction;
    }
  }
  for (size_t i = 0; i < total_mass_fraction.size(); ++i) {
    EKAT_REQUIRE(total_mass_fraction[i] <= 1.0, 
        "Total aerosol mass fraction in mode " << cfg.name << " exceeds 1");
  }

  compute_mass_mixing_ratios_from_mass_fractions(atmosphere, mass_fractions, cfg);
  
  return cfg;
}

AerosolConfig read_aerosols(const AtmosphereConfig &atmosphere, const ekat::ParameterList &params) {
  AerosolConfig cfg = {};
  EKAT_REQUIRE(params.isSublist("modes"), "No aerosol.modes specified");
  auto modes = params.sublist("modes");
  auto mode_names = modes.sublist_names(); // NOTE: sublists interpreted as modes
  for (size_t i = 0; i < mode_names.size(); ++i) {
    auto mode = modes.sublist(mode_names[i]);
    cfg.modes[mode_names[i]] = read_aerosol_mode(atmosphere, mode);
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

GasConfig read_gases(const ekat::ParameterList &params) {
  GasConfig cfg = {};
  auto gas_names = params.param_names();
  for (const std::string &gas_name: gas_names) {
    auto mass_mixing_ratios = read_box_or_column(params, gas_name);
    for (size_t i = 0; i < mass_mixing_ratios.size(); ++i) {
      EKAT_REQUIRE(mass_mixing_ratios[i] >= 0.0,
          "Negative mass mixing ratio at level " << i << " for gas " << gas_name);
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
  cfg.aerosols = read_aerosols(cfg.atmosphere, aerosols);
  cfg.gases = read_gases(aerosols);

  size_t num_modes = cfg.aerosols.modes.size();
  EKAT_REQUIRE(num_modes > 0, "no aerosol modes specified");

  // verify that # of vertical levels is the same everywhere
  size_t n_atm  = cfg.atmosphere.temperature.size();
  size_t n_aero = cfg.aerosols.modes.begin()->second.number_mixing_ratio.size();
  size_t n_gas = cfg.aerosols.modes.size();
  EKAT_REQUIRE(n_atm == n_aero == n_gas,
      "number of vertical levels differs for atmosphere (" << n_atm << "), aerosols (" << n_aero <<
      "), and gases (" << n_gas);

  return cfg;
}

}
