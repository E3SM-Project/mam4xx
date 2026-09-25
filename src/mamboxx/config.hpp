#ifndef MAMBOXX_CONFIG_HPP
#define MAMBOXX_CONFIG_HPP

#include <mam4xx.hpp>

#include <vector>

namespace mamboxx {

using mam4::Real;

struct TimeConfig {
  Real dt;
  int  nstep;
};

struct AtmosphereConfig {
  Real temperature;        // [K]
  Real pressure;           // [Pa]
  Real relative_humidity;  // clear air [-]
};

struct AerosolMassFraction {
  mam4::AeroId species_id;
  Real         value;
};

struct ModeConfig {
  mam4::ModeIndex index;
  Real numc;            // modal number concentration [m-3]
  std::vector<AerosolMassFraction> mass_fractions; // [-]
  Real relative_humidity;  // clear air [-]
};

struct AerosolsConfig {
  std::vector<ModeConfig> modes;
};

struct GasMassMixingRatio {
  mam4::GasId species_id;
  Real        value;
};

struct GasesConfig {
  std::vector<GasMassMixingRatio> mass_mixing_ratios; // [Pa]
};

struct MicrophysConfig {
  bool gas_chemistry;
  bool gas_aer_exch;
  bool rename;
  bool nucleation;
  bool coagulation;
};

struct Config {
  MicrophysConfig microphys;;
  TimeConfig      time;
}

} // namespace mamboxx

#endif
