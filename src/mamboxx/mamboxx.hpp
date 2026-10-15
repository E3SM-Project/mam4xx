// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/mam4_constants.hpp>
#include <mam4xx/mam4_types.hpp>

#include <string>
#include <map>

#ifndef MAMBOXX_HPP
#define MAMBOXX_HPP

namespace mamboxx {

using Real = mam4::Real;

const std::map<std::string, mam4::ModeIndex> mode_indices_for_names = {
  {"accumulation", mam4::ModeIndex::Accumulation},
  {"aitken", mam4::ModeIndex::Aitken},
  {"coarse", mam4::ModeIndex::Coarse},
  {"primary_carbon", mam4::ModeIndex::PrimaryCarbon},
};
const std::map<std::string, mam4::AeroId> aero_ids_for_names = {
  {"bc", mam4::AeroId::BC},
  {"dst", mam4::AeroId::DST},
  {"mom", mam4::AeroId::MOM},
  {"ncl", mam4::AeroId::NaCl},
  {"pom", mam4::AeroId::POM},
  {"so4", mam4::AeroId::SO4},
  {"soa", mam4::AeroId::SOA},
};
const std::map<std::string, mam4::GasId> gas_ids_for_names = {
  {"ccl3f", mam4::GasId::CCl3F},
  {"chcl2f", mam4::GasId::CHCl2F},
  {"ch4", mam4::GasId::CH4},
  {"co2", mam4::GasId::CO2},
  {"dms", mam4::GasId::DMS},
  {"h2o2", mam4::GasId::H2O2},
  {"h2so4", mam4::GasId::H2SO4},
  {"nh3", mam4::GasId::NH3},
  {"n2o", mam4::GasId::N2O},
  {"o2", mam4::GasId::O2},
  {"o3", mam4::GasId::O3},
  {"soag", mam4::GasId::SOAG},
  {"so2", mam4::GasId::SO2},
};

// timestepping configuration
struct TimeConfig {
  Real dt;    // timestep [s]
  int  nstep; // number of timesteps in simulation  [-]
};

// aerosol process switchboard
struct ControlConfig {
  bool gas_chemistry;
  bool gas_aerosol_exchange;
  bool rename;
  bool nucleation;
  bool coagulation;
};

// prevailing atmospheric conditions (column data)
struct AtmosphereConfig {
  std::vector<Real> temperature;        // [K]
  std::vector<Real> pressure;           // [Pa]
  std::vector<Real> relative_humidity;  // for clear air [-]
  // FIXME: add input for vapor_mixing_ratio
  std::vector<Real> vapor_mixing_ratio; // water vapor MMR [kg vapor/kg air]
  Real pblh; // planetary boundary layer height;
};

// aerosol mode (column) data
struct AerosolModeConfig {
  // symbolic name of the mode
  std::string name;
  // modal number mixing ratio [# particles / kg dry air]
  std::vector<Real> number_mixing_ratio;
  // mapping of aerosol species names to mass mixing ratios [kg aerosol / kg dry air]
  std::map<std::string, std::vector<Real>> mass_mixing_ratios;
};

// FIXME: need to distinguish between XConfig (using vectors, maps) and
// FIXME: XState (using views)

struct AerosolConfig {
  // mapping of aerosol mode names to their data
  std::map<std::string, AerosolModeConfig> modes;
  // (can place diagnostics here if needed)
};

struct GasConfig {
  // mapping of gas names to their mass mixing ratios (column data)
  std::map<std::string, std::vector<Real>> mass_mixing_ratios;
};

struct Diagnostics {
};

struct Config {
  TimeConfig       time;
  ControlConfig    control;
  AtmosphereConfig atmosphere;
  AerosolConfig    aerosols;
  GasConfig        gases;
};

struct Result {
  bool          success;
  std::string   error;    // if !success
  AerosolConfig aerosols; // final aerosol state
  GasConfig     gases;    // final gas state
  Diagnostics   diagnostics;
};

// reads a mamboxx config file, returning a Config struct storing its contents
Config read_config(const std::string &filename);

// executes a simulation using the given configuration, returning a result
// indicating success/failure and containing final states
Result run(const Config &config);

// Creates a new column on the device with the given number of vertical levels.
mam4::ColumnView create_column_view(int num_levels);

// Creates a new column on the host with the given number of vertical levels.
using HostColumnView = typename mam4::HostType::view_1d<Real>;
HostColumnView create_host_column_view(int num_levels);

}

#endif
