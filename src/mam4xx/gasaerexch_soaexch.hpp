// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_GASAEREXCH_SOAEXCH_HPP
#define MAM4XX_GASAEREXCH_SOAEXCH_HPP

#include "aero_config.hpp"
#include "aero_modes.hpp"
#include "mam4_math.hpp"

namespace mam4 {
namespace gasaerexch {

// ==============================================================================
// Calculate SOA species's eiquilibrium vapor mixing ratio under the ambient
// condition, ignoring the solute effect.
//
// History:
// - MAM box model by R. C. Easter, PNNL.
// - Subroutine cretated duing code refactoring by Hui Wan, PNNL, 2022.
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void soa_equilib_mixing_ratio_no_solute(const Real &T_in_K,     // in
                                        const Real &p_in_Pa,    // in
                                        const Real &pstd_in_Pa, // in
                                        const Real r_universal, // in
                                        Real &g0_soa) {         // out

  // clang-format off
  // T_in_K         temperature in Kelvin
  // p_in_Pa        air pressure in Pascal
  // pstd_in_Pa     standard air pressure in Pascal
  // r_universal    universal gas constant in J/K/mol
  // g0_soa         ambient soa gas equilib mixing ratio (mol/mol at actual mw)
  // ntot_soaspec   number of SOA species, which is assumed 1 for this function.
  // clang-format on

  // Comment by Hui Wan:
  // Parameters: should they be changed to intent(in)? - Not ncessary for now,
  // but perhaps need to change to intent in when we get multiple SOA species.

  // delh_vap_soa = heat of vaporization for gas soa (J/mol)
  const Real delh_vap_soa = 156.0e3;
  // p0_soa_298 = soa gas equilib vapor presssure (atm) at 298 K
  const Real p0_soa_at_298K = 1.0e-10;

  //
  // Calculate equilibrium mixing ratio under the ambient condition
  //

  // The equilibrium vapor pressure
  const Real p0_soa =
      p0_soa_at_298K * mam4::exp(-(delh_vap_soa / (r_universal / 1.e3)) *
                                 ((1.0 / T_in_K) - (1.0 / 298.0)));

  // Convert to mixing ratio in mol/mol
  g0_soa = pstd_in_Pa * p0_soa / p_in_Pa;
}

// NOTE: This is different from box model.
constexpr int npoa =
    1; //   number of differently tagged primary-organic   aerosol species
constexpr int nsoa =
    1; // number of differently tagged secondary-organic aerosol species
// from: modal_aero_amicphys_control
constexpr int max_mode = AeroConfig::num_modes() + 1;
constexpr int max_gas = nsoa + 1;
//  the +4 in max_aer are dst, ncl, so4, mom
constexpr int nbc =
    1; // number of differently tagged black-carbon      aerosol species
constexpr int max_aer = nsoa + npoa + nbc + 4;
constexpr int naer = nsoa + 1;
constexpr int iaer_pom =
    naer + 1 - 1; // -1 for fotran to c++ indexing conversion.
constexpr int ntot_soamode = 4;

KOKKOS_INLINE_FUNCTION
void mam_soaexch_1subarea(const Real dtsubstep,                 // in
                          const Real temp,                      // in
                          const Real pmid,                      // in
                          Real qgas_cur[max_gas],               // in/out
                          Real qgas_avg[max_gas],               // in/out
                          Real qaer_cur[max_aer][max_mode],     // in/out
                          const Real uptkaer[max_gas][max_mode] // in
) {

  constexpr int ntot_poaspec = npoa;
  constexpr int ntot_soaspec = nsoa;
  int niter_max = 1000;
  int niter = 0;
  const Real dtfull = dtsubstep;
  Real tcur = 0.0;
  Real dtcur = 0.0;
  Real dtsum_qgas_avg = 0.0;
  Real qgas_prv[max_gas] = {0.0};
  Real qaer_prv[max_aer][max_mode] = {{0.0}};
  Real uptkaer_soag_tmp[nsoa][max_mode] = {{0.0}};
  Real a_opoa[max_mode] = {0.0};
  Real a_soa[ntot_soaspec][max_mode] = {{0.0}};
  Real g0_soa[ntot_soaspec] = {0.0};
  Real g_soa[ntot_soaspec] = {0.0};
  Real g_star[ntot_soaspec][max_mode] = {{0.0}};
  Real sat[ntot_soaspec][max_mode] = {{0.0}};

  // BAD CONSTANT
  constexpr Real delh_vap_soa = 156.0e3;
  constexpr Real p0_soa_298 = 1.0e-10;

  // BAD CONSTANT
  Real opoa_frac[ntot_poaspec][max_mode];
  for (int ll = 0; ll < ntot_poaspec; ++ll) {
    for (int n = 0; n < max_mode; ++n) {
      opoa_frac[ll][n] = 0.1;
    }
  }

  // for primary carbon mode, set opoa_frac=0 for consistency with older code
  // (this could be changed)
  constexpr bool flag_pcarbon_opoa_frac_zero = true;
  constexpr int npca = 3; // primary carbon mode number
  if (flag_pcarbon_opoa_frac_zero && npca > 0) {
    for (int ll = 0; ll < ntot_poaspec; ++ll)
      opoa_frac[ll][npca] = 0;
  }

  bool skip_soamode[max_mode];
  for (int n = 0; n < max_mode; ++n) {
    skip_soamode[n] = true;
  }

  constexpr Real alpha_astem = 0.05; // Parameter used in calc of time step
  constexpr Real dtsub_fixed = -1.0; // Fixed sub-step for time integration (s)
  // BAD CONSTANT
  constexpr Real rgas = 8.3144; // gas constant in J/K/mol
  constexpr Real a_min1 = 1.0e-20;
  constexpr Real g_min1 = 1.0e-20;

  Real tot_soa[ntot_soaspec] = {}; // g_soa + sum( a_soa(:) )

  // Calculate ambient equilibrium soa gas
  for (int ll = 0; ll < ntot_soaspec; ++ll) {
    // BAD CONSTANT
    const Real p0_soa = p0_soa_298 * mam4::exp(-(delh_vap_soa / rgas) *
                                               ((1.0 / temp) - (1.0 / 298.0)));
    // BAD CONSTANT
    g0_soa[ll] = 1.01325e5 * p0_soa / pmid;
  }

  // Main integration loop -- does multiple substeps to reach dtfull
  for (int igas = 0; igas < nsoa; ++igas) {
    qgas_avg[igas] = 0.0;
  }
  dtsum_qgas_avg = 0.0;
  // BAD CONSTANT
  while (tcur < dtfull - 1.0e-3) {
    niter++;
    if (niter > niter_max)
      break;

    // Set qxxx_prv to be current value
    for (int igas = 0; igas < nsoa; ++igas) {
      qgas_prv[igas] = qgas_cur[igas];
    }
    for (int iaer = 0; iaer < max_aer; ++iaer) {
      for (int n = 0; n < AeroConfig::num_modes(); ++n) {
        qaer_prv[iaer][n] = qaer_cur[iaer][n];
      }
    }

    // Determine which modes have non-zero transfer rates
    // and are involved in the soa gas-aerosol transfer
    // for diameter = 1 nm and number = 1 #/cm3, xferrate ~= 1e-9 s-1
    for (int n = 0; n < ntot_soamode; ++n) {
      skip_soamode[n] = true;
      for (int ll = 0; ll < ntot_soaspec; ++ll) {
        if (uptkaer[ll][n] > 1.0e-15) {
          uptkaer_soag_tmp[ll][n] = uptkaer[ll][n];
          skip_soamode[n] = false;
        } else {
          uptkaer_soag_tmp[ll][n] = 0.0;
        }
      }
    }

    // Load incoming soag and soaa into temporary arrays
    // force things to be non-negative
    // calc tot_soa(ll)
    // calc a_opoa (always slightly >0)
    //
    // *** questions ***
    // > why not use qgas and qaer instead of g_soa and a_soa
    // > why not calc the following on every substep because
    //      nuc and coag may change things:
    //      skip_soamode, uptkaer_soag_tmp, tot_soa, a_opoa
    // > include gasprod for soa ??
    // > create qxxx_bgn = qxxx_cur at the very beginning (is it needed)
    //
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      g_soa[ll] = mam4::max(qgas_prv[ll], 0.0);
      tot_soa[ll] = g_soa[ll];
      for (int n = 0; n < ntot_soamode; ++n) {
        if (skip_soamode[n])
          continue;
        a_soa[ll][n] = mam4::max(qaer_prv[ll][n], 0.0);
        tot_soa[ll] += a_soa[ll][n];
      }
    }

    for (int n = 0; n < ntot_soamode; ++n) {
      if (skip_soamode[n])
        continue;
      a_opoa[n] = 0.0;
      for (int ll = 0; ll < ntot_poaspec; ++ll) {
        a_opoa[n] +=
            opoa_frac[ll][n] * mam4::max(qaer_prv[iaer_pom + ll][n], 0.0);
      }
    }

    // Determine time step
    Real tmpa = 0.0; // time integration parameter for all soa species
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      Real tmpb = 0.0; // time integration parameter for a single soa species
      for (int n = 0; n < ntot_soamode; ++n) {
        if (skip_soamode[n])
          continue;
        Real a_ooa_sum = a_opoa[n];
        for (int ll = 0; ll < ntot_soaspec; ++ll)
          a_ooa_sum += a_soa[ll][n];
        sat[ll][n] = g0_soa[ll] / mam4::max(a_ooa_sum, a_min1);
        g_star[ll][n] = sat[ll][n] * a_soa[ll][n];
        const Real phi = (g_soa[ll] - g_star[ll][n]) /
                         mam4::max(g_soa[ll], mam4::max(g_star[ll][n], g_min1));
        tmpb += uptkaer_soag_tmp[ll][n] * abs(phi);
      }
      tmpa = mam4::max(tmpa, tmpb);
    }

    if (dtsub_fixed > 0.0) {
      dtcur = dtsub_fixed;
      tcur += dtcur;
    } else {
      Real dtmax = dtfull - tcur;
      if (dtmax * tmpa <= alpha_astem) {
        dtcur = dtmax;
        tcur = dtfull;
      } else {
        dtcur = alpha_astem / tmpa;
        tcur += dtcur;
      }
    }

    // Step 1 - for modes where soa is condensing, estimate "new" a_soa(ll,n)
    // using an explicit calculation with "old" g_soa
    // and g_star(ll,n) calculated using "old" a_soa(ll,n)
    // do this to get better estimate of "new" a_soa(ll,n) and sat(ll,n)
    for (int n = 0; n < ntot_soamode; ++n) {
      if (skip_soamode[n])
        continue;
      Real a_ooa_sum = a_opoa[n];
      for (int ll = 0; ll < ntot_soaspec; ++ll) {
        Real a_soa_sum = a_soa[ll][n];
        const Real beta = dtcur * uptkaer_soag_tmp[ll][n];
        const Real del_g_soa = g_soa[ll] - g_star[ll][n];
        if (del_g_soa > 0.0) {
          a_soa_sum += beta * del_g_soa;
        }
        a_ooa_sum += a_soa_sum;
      }
      for (int ll = 0; ll < ntot_soaspec; ++ll) {
        const Real del_g_soa = g_soa[ll] - g_star[ll][n];
        if (del_g_soa > 0.0) {
          sat[ll][n] = g0_soa[ll] / mam4::max(a_ooa_sum, a_min1);
        }
      }
    }

    // Step 2 - implicit in g_soa and semi-implicit in a_soa,
    // with g_star(ll,n) calculated semi-implicitly
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      tmpa = 0.0;
      Real tmpb = 0.0;
      for (int n = 0; n < ntot_soamode; ++n) {
        if (skip_soamode[n])
          continue;
        const Real beta = dtcur * uptkaer_soag_tmp[ll][n];
        tmpa += a_soa[ll][n] / (1.0 + beta * sat[ll][n]);
        tmpb += beta / (1.0 + beta * sat[ll][n]);
      }

      g_soa[ll] = (tot_soa[ll] - tmpa) / (1.0 + tmpb);
      g_soa[ll] = mam4::max(0.0, g_soa[ll]);
      for (int n = 0; n < ntot_soamode; ++n) {
        if (skip_soamode[n])
          continue;
        const Real beta = dtcur * uptkaer_soag_tmp[ll][n];
        a_soa[ll][n] =
            (a_soa[ll][n] + beta * g_soa[ll]) / (1.0 + beta * sat[ll][n]);
      }
    }

    // Update mix ratios for soa species
    for (int igas = 0; igas < nsoa; ++igas) {
      int iaer = igas;
      qgas_cur[igas] = g_soa[igas];
      const Real tmpc = qgas_cur[igas] - qgas_prv[igas];
      qgas_avg[igas] += dtcur * (qgas_prv[igas] + 0.5 * tmpc);
      for (int n = 0; n < ntot_soamode; ++n) {
        qaer_cur[iaer][n] = a_soa[iaer][n];
      }
    }

    dtsum_qgas_avg += dtcur;
  }

  // Convert qgas_avg from sum_over[ qgas*dt_cut ] to an average
  for (int igas = 0; igas < nsoa; ++igas) {
    qgas_avg[igas] = mam4::max(0.0, qgas_avg[igas] / dtsum_qgas_avg);
  }
}
//================================================================================
// Determine adaptive time-step size for SOA condensation/evaporation
//================================================================================
KOKKOS_INLINE_FUNCTION
Real soa_exch_substepsize(
    const int ntot_soamode,    // in
    const int ntot_soaspec,    // in
    const bool skip_soamode[], // in len ntot_soaspec
    const Real uptkaer_soag_tmp[]
                               [AeroConfig::num_modes()], // in len ntot_soaspec
    const Real a_soa[][AeroConfig::num_modes()],          // in len ntot_soaspec
    const Real a_opoa[AeroConfig::num_modes()],           // in
    const Real g_soa[],                                   // in len ntot_soaspec
    const Real g0_soa[],                                  // in len ntot_soaspec
    const Real alpha_astem,                               // in
    const Real dt_full,                                   // in
    Real &t_cur)                                          // inout
{
  // clang-format off
  // ntot_soaspec
  //  "last" gas species that can be SOA, MAM4 had this variable but it was
  //  a work in progress and not suported in mam4xx but it was desired to
  //  keep these variables around to help support variable number of SOA
  //  species in the future.
  //
  // ntot_soamode  "last" mode on which soa is allowed to condense
  //
  // skip_soamode          true if this mode does not have soa
  // uptkaer_soag_tmpmodes evolving SOA aerosol mixrat (mol/mol at actual mw),
  //                       part of the unknowns of the ODEs
  // a_soa                 soa aerosol mixrat (mol/mol at actual mw)
  // a_opoa                oxidized-poa aerosol mixrat (mol/mol at actual mw)
  // g_soa                 evolving SOA gas mixrat (mol/mol at actual mw),
  //                       part of the unknowns of the ODEs
  // g0_soa                ambient soa gas equilib mixrat (mol/mol at actual mw),
  //                       length based on ntot_soaspec=1
  // alpha_astem           error control parameter for sub-timesteps
  // dt_fulln              host model dt (s)
  // t_cur                 current time (s) since last full time step
  // clang-format on

  const Real eps_aer = 1.0e-20; // epsilon to be used on denominator for
                                // avoiding division by zero
  const Real eps_gas = 1.0e-20; // epsilon to be used on denominator for
                                // avoiding division by zero

  //------------------------------------------------------------------
  // Oxygenated OA in each aerosol mode, summed over all SOA species
  //------------------------------------------------------------------
  // total ooa (=soa+opoa) in a mode
  Real a_ooa_sum_tmp[AeroConfig::num_modes()];
  for (int n = 0; n < ntot_soamode; ++n) {
    if (!skip_soamode[n]) {
      a_ooa_sum_tmp[n] = a_opoa[n];
      for (int i = 0; i < ntot_soaspec; ++i)
        a_ooa_sum_tmp[n] += a_soa[i][n];
    }
  }
  //----------------------------------------------------------------------------------------------------------
  // For each SOA species, estimate the normalized total mass exchange rate by
  // using a simple Euler forward scheme.  Here the normalized mass exchange
  // rate is defined as
  //   condensed or evaporated SOA /dt / mam4::max( available-SOA-gas,
  //   equilibrium-SOA-gas-mixing-ratio )
  // We first calculate this rate for each SOA species and each aerosol mode.
  // Then, for each species, we calculate the sum of its absolute value over all
  // modes, and saved the result in the array tot_frac_single_soa_species.
  //----------------------------------------------------------------------------------------------------------
  Real tot_frac_single_soa_species[AeroConfig::num_gas_ids()] = {0};
  for (int i = 0; i < ntot_soaspec; ++i)
    tot_frac_single_soa_species[i] = 0.0;

  for (int ll = 0; ll < ntot_soaspec; ++ll) {
    for (int n = 0; n < ntot_soamode; ++n) {
      if (!skip_soamode[n]) {
        // Calculate g_star, the equilibrium SOA mixing ratio after taking into
        // account the solute effect

        // sat(ll,n) = g0_soa(ll)/a_ooa_sum_tmp(n) = g_star(ll,n)/a_soa(ll,n)
        // This this the pre-factor of SOA (aerosol) mixing ratio in the
        // expression of the equilibrium SOA mixing ratio -- it is not a
        // saturation rato!
        const Real sat = g0_soa[ll] / mam4::max(a_ooa_sum_tmp[n], eps_aer);

        // g_star -  equilibrium SOA gas mixing ratio (mol/mol at actual mw),
        // with solute effect accounted for
        const Real g_star = sat * a_soa[ll][n];

        // Calculate the mass exchange rate (normalized by the larger between
        // g_soa and g_star) of this aerosol mode by using an Euler forward
        // scheme, then add this fraction to the total fraction (variable
        // tot_frac_single_soa_species) summed over all aerosol modes.

        const Real phi = (g_soa[ll] - g_star) /
                         mam4::max(g_soa[ll], mam4::max(g_star, eps_gas));
        tot_frac_single_soa_species[ll] += uptkaer_soag_tmp[ll][n] * abs(phi);
      }
    }
  }

  //------------------------------------------------------------------------------------------------------
  // Now, get the maximum of the normalized exchange rate of all SOA species
  //------------------------------------------------------------------------------------------------------
  Real max_frac_all_soa_species = tot_frac_single_soa_species[0];
  for (int i = 0; i < ntot_soaspec; ++i)
    max_frac_all_soa_species =
        mam4::max(max_frac_all_soa_species, tot_frac_single_soa_species[i]);

  //------------------------------------------------------------------------------------------------------
  // Determine the adaptive step size, dt_cur, requiring that the condensed or
  // evaporated amount of any SOA species cannot exceed a fraction (alpha_astem)
  // of
  //   mam4::max( available-SOA-gas, equilibrium-SOA-gas-mixing-ratio ).
  //------------------------------------------------------------------------------------------------------

  // how far we are from the end of the full time step
  const Real dt_max = dt_full - t_cur;

  // out current integration timestep (s)
  Real dt_cur = 0;
  if (dt_max * max_frac_all_soa_species <= alpha_astem) {
    // Here alpha_astem/max_frac_all_soa_species >= dt_max, so this is the final
    // substep. The substep length is limited by the length of full dt.
    dt_cur = dt_max;
    t_cur = dt_full;
  } else {
    // Using dt_max would lead to more than alpha_astem (fraction) of mass
    // exchange. Limit the step size to avoid exceeding alpha_astem.
    dt_cur = alpha_astem / max_frac_all_soa_species;
    t_cur += dt_cur;
  }
  return dt_cur;
}
//===============================================================================================

//===============================================================================================
// Time integration for the ODE set that describes the condensation/evaporation
// of SOA
//-----------------------------------------------------------------------------------------------
// History:
// - Original version from the MAM box model by R. C. Easter, PNNL.
// - Refactoring by Hui Wan and Qiyang Yan, PNNL, 2022.
//===============================================================================================
KOKKOS_INLINE_FUNCTION
void mam_soaexch_advance_in_time(
    const int ntot_soamode,    // in
    const int ntot_soaspec,    // in
    const GasId soaspec[],     // in len ntot_soaspec
    const AeroId gas_to_aer[], // in
    const Real dt_full,        // in
    const Real dt_sub_fixed,   // in
    const int niter_max,       // in
    const Real alpha_astem,    // in
    const Real uptkaer[AeroConfig::num_gas_ids()]
                      [AeroConfig::num_modes()], // in
    const Real g0_soa[],                         // in len ntot_soaspec
    Real qgas_cur[AeroConfig::num_gas_ids()],    // inout
    const Real a_opoa[AeroConfig::num_modes()],  // in
    Real qaer_cur[AeroConfig::num_aerosol_ids()]
                 [AeroConfig::num_modes()],   // inout
    Real qgas_avg[AeroConfig::num_gas_ids()], // inout
    int &niter)                               // out
{
  // clang-format off
  // int ntot_soamode
  // int ntot_soaspec  "last" gas species that can be SOA
  // int soaspec[ntot_soaspec]
  // const AeroId gas_to_aer[GasAerExch::num_gas],
  // dt_full       Host model dt (s)
  // dt_sub_fixed  Fixed sub-step. A negative value means using adaptive step sizes
  // niter_max     Maximum number of substeps
  // alpha_astem   Error control parameter for sub-timesteps
  // uptkaer       Uptake rate coefficient
  // g0_soa        Ambient soa gas equilib mixrat (mol/mol at / actual mw),
  //               len ntot_soaspec=1 since multiple soa not currently supported
  // qgas_cur
  // a_opoa        Oxidized-poa aerosol mixrat (mol/mol at actual mw)
  // qaer_cur
  // qgas_avg
  // niter         Total number of sub-steps
  // clang-format on

  static constexpr int max_mode = AeroConfig::num_modes();

  // The local arrays a_soa and g_soa declared below have the same meaning of
  // qaer_cur and qgas_cur.
  // - At the beginning of each substep, a_soa and g_soa get their values from
  // qaer_cur and qgas_cur but
  //   with negative values set to zero.
  //- At the end of each substep, the values of a_soa and g_soa are copied back
  // to qaer_cur and qgas_cur.

  // evolving SOA aerosol mixrat (mol/mol at actual mw), part of the unknowns of
  // the ODEs
  Real a_soa[AeroConfig::num_gas_ids()][max_mode] = {};
  // evolving SOA gas mixrat (mol/mol at actual mw), part of the unknowns of the
  // ODEs
  Real g_soa[AeroConfig::num_gas_ids()] = {};

  // gas mixrat at beginning of substep, saved for calculating time average of
  // qgas
  Real qgas_prv[AeroConfig::num_gas_ids()] = {};
  // qgas*dt integrated over a time step, used for calculating time average of
  // qgas
  Real qgas_avg_sum[AeroConfig::num_gas_ids()] = {};

  // true if this mode does not have soa
  bool skip_soamode[max_mode] = {};
  // uptake rate of different modes for different soa species
  Real uptkaer_soag_tmp[AeroConfig::num_gas_ids()][max_mode] = {};
  // dt_cur * uptake-rate-coefficient
  Real beta[AeroConfig::num_gas_ids()][max_mode] = {};

  const Real eps_aer = 1.0e-20; // epsilon to be used on denominator for
                                // avoiding division by zero
  const Real eps_dt = 1.0e-3;

  // variable name "sat": sat(m,ll) = g0_soa(ll)/a_ooa_sum(m) =
  // g_star(m,ll)/a_soa(m,ll) used by the numerical integration scheme -- it is
  // not a saturation rato!

  Real sat_hybrid[AeroConfig::num_gas_ids()][max_mode] = {};

  Real tot_soa[AeroConfig::num_gas_ids()] = {}; // g_soa + sum( a_soa(:) )

  // ----------------------------------------------------------------------
  //  Determine which modes have non-zero transfer rates and are hence
  //  involved in the subsequent calculations of soa gas-aerosol transfer.
  //  (For diameter = 1 nm and number = 1 #/cm3, xferrate ~= 1e-9 s-1)
  // ----------------------------------------------------------------------
  for (int n = 0; n < max_mode; ++n)
    skip_soamode[n] = true;

  for (int n = 0; n < ntot_soamode; ++n) {
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      const int soa = static_cast<int>(soaspec[ll]);
      if (uptkaer[soa][n] > 1.0e-15) {
        uptkaer_soag_tmp[ll][n] = uptkaer[soa][n];
        skip_soamode[n] = false;
      } else {
        uptkaer_soag_tmp[ll][n] = 0.0;
      }
    }
  }
  // -------------------------------------
  //  Time loop for SOA sub-stepping
  // -------------------------------------
  niter = 0;
  // time elapsed since the beginning of sub-cycling (last full timestep)
  Real tcur = 0.0;

  // total dt of all substeps, used for calculating time average of qgas
  Real dtsum_qgas_avg = 0.0;
  for (int i = 0; i < ntot_soaspec; ++i)
    qgas_avg_sum[i] = 0.0;

  while (tcur < dt_full - eps_dt) {
    ++niter;
    if (niter > niter_max)
      break;
    // save gas mixing ratios at the beginning of substep.
    // This is used at the end of the substep to calculate a time average
    for (int i = 0; i < ntot_soaspec; ++i) {
      const int soa = static_cast<int>(soaspec[i]);
      qgas_prv[i] = qgas_cur[soa];
    }

    // ------------------------------------------------------------------------------
    //  SOA gas, aerosol, and total: get the current (old) values
    // ------------------------------------------------------------------------------
    //  Load incoming SOA gas into temporary array, force values to be
    //  non-negative
    for (int i = 0; i < ntot_soaspec; ++i) {
      const int soa = static_cast<int>(soaspec[i]);
      g_soa[i] = mam4::max(qgas_cur[soa], 0.0);
    }

    //  Load incoming SOA aerosols into temporary array, force values to be
    //  non-negative
    for (int n = 0; n < ntot_soamode; ++n) {
      if (!skip_soamode[n]) {
        for (int i = 0; i < ntot_soaspec; ++i) {
          const int soa =
              static_cast<int>(gas_to_aer[static_cast<int>(soaspec[i])]);
          a_soa[i][n] = mam4::max(qaer_cur[soa][n], 0.0);
        }
      }
    }

    //  Calculate the total amount of each SOA species, defined as mixing ratio
    //  of gas + aerosol, saved in array tot_soa.
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      tot_soa[ll] = g_soa[ll];
      for (int n = 0; n < ntot_soamode; ++n) {
        if (!skip_soamode[n]) {
          tot_soa[ll] += a_soa[ll][n];
        }
      }
    }

    // -----------------------------------------------------------------------------
    //  Determine step size: either fixed or adaptive
    // -----------------------------------------------------------------------------
    Real dt_cur = 0;          // substep length, fixed or adaptive
    if (dt_sub_fixed > 0.0) { //  Use prescribed, fixed step size
      dt_cur = dt_sub_fixed;
      tcur += dt_cur;
    } else {
      // Choose an adaptive step size
      dt_cur = soa_exch_substepsize(ntot_soamode, ntot_soaspec, skip_soamode,
                                    uptkaer_soag_tmp, a_soa, a_opoa, g_soa,
                                    g0_soa, alpha_astem, dt_full, tcur);
    }

    // ----------------------------------------------------------------------------
    //  Now that we have determined the sub-step size dt_cur,
    //  define a variable beta = dt_cur * uptake-rate-coefficient.
    //  This is used at a few places in the semi-implicit time integration
    //  scheme.
    // ----------------------------------------------------------------------------
    for (int n = 0; n < ntot_soamode; ++n) {
      if (!skip_soamode[n]) {
        for (int ll = 0; ll < ntot_soaspec; ++ll)
          beta[ll][n] = dt_cur * uptkaer_soag_tmp[ll][n];
      }
    }

    // ------------------------------------------------------------------------------------------
    //  Linearize the ODE of each SOA species in each mode
    // ------------------------------------------------------------------------------------------
    //  Because the equilibrium SOA mixing ratio (that takes into account the
    //  solvent effect) depends on the SOA (aerosol) mixing ratio in each mode,
    //  the time evolution equations for SOAs (aerosol) are nonlinear.
    //  To numerically solve these nonlinear equations using a
    //  semi-implicit-in-time scheme, we need to linearize the equations. The
    //  nonlinear pre-factor in front of an SOA mixing ratio on the RHS of the
    //  SOA mixing ratio equation is
    //      uptake-rate-coefficient * g0_soa / mixing-ratio-of-OOA
    //  Let us denote
    //      sat = g0_soa / mixing-ratio-of-OOA
    //  The code block below provides an approximate value of sat (saved in the
    //  array sat_hybrid) to be used in the semi-implicit solve further down
    //  below. We refer to this "sat" variable as a "hybrid" one because the
    //  calculation below uses different expressions for SOA condensation and
    //  evaporation. The difference is in the SOA (aerosol) mixing ratios used
    //  for calculating the OOA mixing ratio in the denominaotor of sat.
    // ------------------------------------------------------------------------------------------

    // temporary SOA aerosol mixrat (mol/mol) used for linearization
    Real a_soa_hybrid[AeroConfig::num_gas_ids()];
    for (int n = 0; n < ntot_soamode; ++n) {
      if (skip_soamode[n])
        continue;

      for (int ll = 0; ll < ntot_soaspec; ++ll) {

        // First get an estimate of the equilibrium SOA mixing ratio (variable
        // g_star_old) using the old SOA mixing ratio (variable a_soa)

        // total ooa (=soa+opoa) in a mode, calculated using old SOA mixing
        // ratio
        Real a_ooa_sum_old = a_opoa[n];
        for (int i = 0; i < ntot_soaspec; ++i)
          a_ooa_sum_old += a_soa[i][n];
        const Real sat_old = g0_soa[ll] / mam4::max(a_ooa_sum_old, eps_aer);

        // soa gas mixrat that is in equilib with each aerosol mode (mol/mol)
        // diagnosed using old gas and aerosol mixing ratios
        const Real g_star_old = sat_old * a_soa[ll][n];

        //  Using g_star_old and the current (old) g_soa to determine whether we
        //  have
        //   - supersaturation (meaning SOA will be condensing) or
        //   - undersaturation (meaning SOA will be evaporating)

        // SOA gas supersaturation mixrat (mol/mol at actual mw). < 0 means
        // unsaturated
        const Real g_soa_supersat = g_soa[ll] - g_star_old;

        if (g_soa_supersat > 0.0) {
          //  For modes where SOA is condensing, estimate an approximate "new"
          //  a_soa(ll,n) using the Euler forward scheme in which both the SOA
          //  gas mixing ratio and equilibrium mixing ratio are set to their
          //  "old" values, i.e., the current g_soa and the above-calculated
          //  g_star_old. Do this to get better estimate of "new" a_soa(ll,n)
          //  and g_star(ll,n)

          a_soa_hybrid[ll] = a_soa[ll][n] + beta[ll][n] * g_soa_supersat;

        } else {
          //  For modes where SOA is evaporating, simply use the "old" SOA
          //  mixing ratio
          a_soa_hybrid[ll] = a_soa[ll][n];
        }
      }

      //  Now, calculate the total OOA in the mode using the a_soa_hybrid
      //  calculated just now

      // total ooa (=soa+opoa) in a mode, different for condensation/evaporation
      // cases
      Real a_ooa_sum_hybrid = a_opoa[n];
      for (int i = 0; i < ntot_soaspec; ++i)
        a_ooa_sum_hybrid += a_soa_hybrid[i];

      //  With the newly calculated a_ooa_sum_hybrid, we can now calculate
      //  the pre-factor in front of the SOA mixing ratio on the RHS of each SOA
      //  equation, (i.e., variable sat_hybrid).

      for (int ll = 0; ll < ntot_soaspec; ++ll)
        sat_hybrid[ll][n] = g0_soa[ll] / mam4::max(a_ooa_sum_hybrid, eps_aer);
    }
    // ------------------------------------------------------------------------------------------
    //  Implicit solve for the linearize equations
    // ------------------------------------------------------------------------------------------
    for (int ll = 0; ll < ntot_soaspec; ++ll) {
      Real tmpa = 0.0;
      Real tmpb = 0.0;
      for (int n = 0; n < ntot_soamode; ++n) {
        if (!skip_soamode[n]) {
          tmpa += a_soa[ll][n] / (1.0 + beta[ll][n] * sat_hybrid[ll][n]);
          tmpb += beta[ll][n] / (1.0 + beta[ll][n] * sat_hybrid[ll][n]);
        }
      }
      g_soa[ll] = (tot_soa[ll] - tmpa) / (1.0 + tmpb);
      g_soa[ll] = mam4::max(0.0, g_soa[ll]);
      for (int n = 0; n < ntot_soamode; ++n) {
        if (!skip_soamode[n]) {
          a_soa[ll][n] = (a_soa[ll][n] + beta[ll][n] * g_soa[ll]) /
                         (1.0 + beta[ll][n] * sat_hybrid[ll][n]);
        }
      }
    }

    // ------------------------------------------------------------------------------------------
    //  Save mix ratios for soa species
    // ------------------------------------------------------------------------------------------
    for (int igas = 0; igas < ntot_soaspec; ++igas) {
      const int soa =
          static_cast<int>(gas_to_aer[static_cast<int>(soaspec[igas])]);
      for (int n = 0; n < ntot_soamode; ++n) {
        qaer_cur[soa][n] = a_soa[igas][n];
      }
    }

    // ------------------------------------------------------------------------------------------
    //  Save mixing ratios for SOA gas species; diagnose time average
    // ------------------------------------------------------------------------------------------
    for (int igas = 0; igas < ntot_soaspec; ++igas) {
      const int soa = static_cast<int>(soaspec[igas]);
      qgas_cur[soa] = g_soa[igas]; //  new gas mixing ratio
      const Real tmpc =
          qgas_cur[soa] - qgas_prv[igas]; //  amount of condensation/evaporation
      qgas_avg_sum[igas] =
          qgas_avg_sum[igas] + dt_cur * (qgas_prv[igas] + 0.5 * tmpc);
    }
    dtsum_qgas_avg = dtsum_qgas_avg + dt_cur;
  }

  // -------------------------------------------------------------------
  //  Convert qgas_avg from sum_over[ qgas*dt_cur ] to an average
  // -------------------------------------------------------------------
  for (int i = 0; i < ntot_soaspec; ++i) {
    const int soa = static_cast<int>(soaspec[i]);
    qgas_avg[soa] = mam4::max(0.0, qgas_avg_sum[i] / dtsum_qgas_avg);
  }
}

// --------------------------------------------------------------------
// Calculate secondary organic aerosols, soa,
// condensation/evaporation over time dt
// --------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void mam_soaexch_1subarea(const int mode_pca,          // in
                          const int ntot_soamode,      // in
                          const int ntot_soaspec,      // in
                          const GasId soaspec[],       // in len ntot_soaspec
                          const AeroId gas_to_aer[],   // in
                          const Real dt,               // in
                          const Real dt_sub_soa_fixed, // in
                          const Real pstd,             // in
                          const Real r_universal,      // in
                          const Real temp,             // in
                          const Real pmid,             // in
                          const Real uptkaer[AeroConfig::num_gas_ids()]
                                            [AeroConfig::num_modes()],     // in
                          const Real qaer_poa[1][AeroConfig::num_modes()], // in
                          Real qgas_cur[AeroConfig::num_gas_ids()], // inout
                          Real qgas_avg[AeroConfig::num_gas_ids()], // inout
                          Real qaer_cur[AeroConfig::num_aerosol_ids()]
                                       [AeroConfig::num_modes()], // inout
                          int &niter,                             // out
                          Real &g0_soa) {                         // out

  // clang-format off
  // mode_pca         mam4::ModeIndex::PrimaryCarbon
  // ntot_soamode
  // ntot_soaspec
  // soaspec[ntot_soaspec]
  // const AeroId gas_to_aer[GasAerExch::num_gas],
  // dt               time step size used by parent subroutine
  // dt_sub_soa_fixed fixed sub-step in s. A negative value  means using adaptive step sizes
  // pstd             standard atmosphere in Pa
  // r_universal      universal gas constant in J/K/mol
  // temp             temperature (K)
  // pmid             pressure at model levels (Pa)
  // uptkaer          uptake rate
  // qaer_poa         POA mixing ratio (mol/mol at actual mw)
  // qgas_cur         current gas mixing ratio
  // qgas_avg
  // qaer_cur         current aerosol mass mix ratio (mol/mol)
  // niter
  // g0_soa           ambient soa gas equilib mixrat (mol/mol at actual mw)
  // clang-format on

  // ntot_poaspec is the number of reacting gas species. The code only supports
  // one but MAM4 had an initial support for more so it was decided to keep the
  // form of the multi-species code if not the function.
  static constexpr int ntot_poaspec = 1;
  static constexpr int num_mode = AeroConfig::num_modes();
  // for backward compatibility
  static constexpr bool flag_pcarbon_opoa_frac_zero = true;
  // parameter used in calc of sub-step sizes
  static constexpr Real alpha_astem = 0.05;
  // assumed OPOA fraction
  static constexpr Real opoa_frac_const = 0.10;
  Real a_opoa[num_mode];
  for (int n = 0; n < num_mode; ++n)
    a_opoa[n] = 0;
  // ----------------------------------------------------------------------
  // SOA species's equilibrium vapor mixing ratio at the ambient T and p,
  // assuming no solute effect. (This equilibrium mixing ratio depends only on
  // T, p and the parameters used inside the subroutine. There is no dependence
  // on SOA gas or aerosol mixing ratios, so the calculation does not need to
  // repeat during temporal sub-cycling.)
  // -----------------------------------------------------------------------
  soa_equilib_mixing_ratio_no_solute(temp, pmid, pstd, r_universal, g0_soa);
  // ------------------------------------------------------------------------------------
  // Calculate the mixing ratio of oxygeneated POA (OPOA) in each mode.
  // (This is needed to account for the solute effect on SOA's equilibrium
  // mixing ratio.)
  // --------------------------------------------------------------------------------------
  // Use the assumed OPOA fraction, per POA species and mode.
  Real opoa_frac[ntot_poaspec][num_mode];
  for (int i = 0; i < ntot_poaspec; ++i)
    for (int j = 0; j < num_mode; ++j)
      opoa_frac[i][j] = opoa_frac_const;

  // For primary carbon mode, keep the option to set opoa_frac=0 for consistency
  // with older code
  if (flag_pcarbon_opoa_frac_zero) {
    for (int i = 0; i < ntot_poaspec; ++i)
      opoa_frac[i][mode_pca] = 0.0;
  }

  // Within each mode, sum up the OPOA mixing ratios of all POA species.
  // The calculation uses the oxygenation fraction specified above and the
  // current (old) POA mixing ratios.

  for (int n = 0; n < ntot_soamode; ++n) {
    for (int i = 0; i < ntot_poaspec; ++i)
      a_opoa[n] += opoa_frac[i][n] * qaer_poa[i][n];
  }
  // -----------------------------------------------------------
  //  Time stepping -- uses multiple substeps to reach dtfull
  // -----------------------------------------------------------
  const int niter_max = 1000;

  mam_soaexch_advance_in_time(ntot_soamode, ntot_soaspec, soaspec, gas_to_aer,
                              dt, dt_sub_soa_fixed, niter_max, alpha_astem,
                              uptkaer, &g0_soa, qgas_cur, a_opoa, qaer_cur,
                              qgas_avg, niter);
}
} // namespace gasaerexch
} // namespace mam4
#endif
