// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4_config.hpp>

#include <string>
#include <map>

#ifndef MAMBOXX_HPP
#define MAMBOXX_HPP

namespace mamboxx {

using Real = mam4::Real;

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
  std::vector<Real> temperature;       // [K]
  std::vector<Real> pressure;          // [Pa]
  std::vector<Real> relative_humidity; // for clear air [-]
};

// aerosol mode (column) data
struct AerosolModeConfig {
  // symbolic name of the mode
  std::string name;
  // modal number concentration
  std::vector<Real> numc;
  // mapping of aerosol species names to mass fractions
  std::map<std::string, std::vector<Real>> mass_fractions;
};

struct AerosolState {
  // mapping of aerosol mode names to their data
  std::map<std::string, AerosolModeConfig> modes;
  // (can place diagnostics here if needed)
};

struct GasState {
  // mapping of gas names to their mass mixing ratios (column data)
  std::map<std::string, std::vector<Real>> mass_mixing_ratios;
};

struct Config {
  TimeConfig       time;
  ControlConfig    control;
  AtmosphereConfig atmosphere;
  AerosolState     aerosols;
  GasState         gases;
};

struct Result {
  bool         success;
  std::string  error;    // if !success
  AerosolState aerosols; // final aerosol state
  GasState     gases;    // final gas state
};

// reads a mamboxx config file, returning a Config struct storing its contents
Config read_config(const std::string &filename);

// executes a simulation using the given configuration, returning a result
// indicating success/failure and containing final states
Result run(const Config &config);

}

#endif
