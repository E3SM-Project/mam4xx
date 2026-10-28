// Copyright (c) 2021, National Technology & Engineering Solutions of Sandia,
// LLC (NTESS). Copyright (c) 2022, Battelle Memorial Institute
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_PHYSICAL_CONSTANTS_HPP
#define MAM4XX_PHYSICAL_CONSTANTS_HPP

#include <mam4xx/mam4_config.hpp>

// clang-format off
// All physical constants in mam4xx are in SI units unless it's explicitly
// stated otherwise. Units given in brackets are SI units and are included
// here only for clarity.
//
// IMPLEMENTATION NOTE: we store constants in a struct so their static
// declarations work in a CUDA environment.

namespace mam4 {

struct Constants {

/** @defgroup Physical_constants_double_precision_required Physical constants (double_precision)

    Force-double precision for constants in this group.

    @{
*/
/// pi
static constexpr double pi = 3.14159265358979323846264;
/// 1/6th of the pi (required for lognormally distributed particle size/volume calculations)
static constexpr double pi_sixth = pi/6;
/// Avogadro's constant [#/mol]
static constexpr double avogadro = 6.022214076e23;
/// Boltzmann's constant [J/K]
static constexpr double boltzmann = 1.380649e-23;
/// @}

/** @defgroup Physical_constants_single_precision_allowed Physical constants (working_precision)

    Configurable precision (single or double) constants.

    @{
*/
/// Universal gas constant [J/(K mol)]
static constexpr Real r_gas = avogadro*boltzmann;

/// acceleration of gravity [m/s^2]
static constexpr Real gravity = 9.80616;

/// Molecular weight of water [kg/mol]
static constexpr Real molec_weight_h2o = 0.018016;

/// Molecular weight of carbon [kg/mol]
static constexpr Real molec_weight_c = 0.0120107;

/// Molecular weight of sodium chloride [kg/mol]
static constexpr Real molec_weight_nacl = 0.05844;

/// Molecular weight of dry air [kg/mol]
static constexpr Real molec_weight_dry_air = 0.028966;

/// Molecular weight of sulfuric acid @f$\text{H}_2\text{SO}_4@f$ [kg/mol]
static constexpr Real molec_weight_h2so4 = 0.098079;

/// Accommodation coefficient (for condensation) value for h2so4 []
static constexpr Real accom_coef_h2so4 = 0.65;

/// Molecular diffusion volume of h2so4 [-]
static constexpr Real molec_diffusion_h2so4 = 42.88;

/// Ratio of gas uptake coeff for SOAG w.r.t. that of h2so4 []
static constexpr Real soag_h2so4_uptake_coeff_ratio = 0.81;

/// Ratio of gas uptake coeff for NH3 w.r.t. that of h2so4 []
static constexpr Real nh3_h2so4_uptake_coeff_ratio = 2.08;

/// Molecular weight of ammonia @f$\text{N}\text{H}_3@f$ [kg/mol]
static constexpr Real molec_weight_nh3 = 0.01703052;

/// Molecular weight of sulfate ion @f$\text{SO}_4^{2-}@f$ [kg/mol]
static constexpr Real molec_weight_so4 = 0.09606;

/// Molecular weight of ammonium ion @f$\text{NH}_4^+@f$ [kg/mol]
static constexpr Real molec_weight_nh4 = 0.018039;

/// Mass density of water [kg/m^3]
static constexpr Real density_h2o = 1.0e3;

/// Pressure at standard conditions (STP) [Pa]
static constexpr Real pressure_stp = 101325.0;

/// Freezing point of water [K]
static constexpr Real freezing_pt_h2o = 273.15;

/// Triple point of water [K]
static constexpr Real triple_pt_h2o = 273.16;

/// Melting point of water [K]
static constexpr Real melting_pt_h2o = freezing_pt_h2o;

/// Boiling point of water [K]
static constexpr Real boil_pt_h2o = 373.16;

/// Molecular diffusion volume of dry air [-]
static constexpr Real molec_diffusion_dry_air = 20.1;

/// Water-to-dry-air weight ratio [-]
static constexpr Real weight_ratio_h2o_air = molec_weight_h2o / molec_weight_dry_air;

/// Latent heat of evaporation [J/kg]
static constexpr Real latent_heat_evap = 2.501e6;

/// Latent heat of fustion [J/kg]
static constexpr Real latent_heat_fusion = 3.337e5;

/// Water vapor gas constant [J/K/kg]
static constexpr Real r_gas_h2o_vapor = r_gas / molec_weight_h2o;

/// Dry air gas constant [J/K/kg]
static constexpr Real r_gas_dry_air = r_gas / molec_weight_dry_air;

/// Specific heat (at constant pressure) of dry air [J/kg/K]
static constexpr Real cp_dry_air = 1.00464e3;

/// Surface tension at water-air interface at 273K [N/m]
static constexpr Real surface_tension_h2o_air_273k = 0.07564;

/// Dry adiabatic lapse rate [K/m]
static constexpr Real dry_adiabatic_lapse_rate = 0.0098;

/// Critical temperature of water [K]
static constexpr Real tc_water = 647.096;

/// @}

}; // struct Constants

} // namespace mam4
// clang-format on
#endif
