// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_GASAEREXCH_HPP
#define MAM4XX_GASAEREXCH_HPP

#include "aero_config.hpp"
#include "aero_modes.hpp"
#include "atmosphere.hpp"
#include "gasaerexch_soaexch.hpp"
#include "mam4_constants.hpp"
#include "mam4_math.hpp"
#include "mam4_types.hpp"
#include "surface.hpp"

#include <Kokkos_Array.hpp>

namespace mam4 {

/// @class GasAerExch
/// This class implements MAM4's gas/aersol exchange parameterization.
class GasAerExch {
public:
  static constexpr int num_mode = AeroConfig::num_modes();

  KOKKOS_INLINE_FUNCTION
  static const ModeIndex (&Modes())[num_mode] {
    static const ModeIndex modes[num_mode] = {
        ModeIndex::Accumulation, ModeIndex::Aitken, ModeIndex::Coarse,
        ModeIndex::PrimaryCarbon};
    return modes;
  }

  // In MAM4, there are only two gases that condense to aerosols:
  // 1. H2SO4 -> SO4
  // 2. SOAG  -> SOA
  KOKKOS_INLINE_FUNCTION static constexpr AeroId gas_to_aer(const GasId gas) {
    AeroId air = AeroId::None;
    if (GasId::H2SO4 == gas)
      air = AeroId::SO4;
    else if (GasId::SOAG == gas)
      air = AeroId::SOA;
    return air;
  }

  // -----------------------------------------------------------------
  // The NA, ANAL, and IMPL flags are used in the eqn_and_numerics_category
  // array to flag how to deal with gas-aerosol mass exchange.
  //
  // MAM currently uses a splitting method to deal with gas-aerosol
  // mass exchange. A quasi-analytical solution assuming time step-wise
  // constant uptake rate is applied to nonvolatile species, while
  // an implicit time stepping method with adaptive step size
  // is applied to gas-phase SOA species which are assumed semi-volatile.
  // There are two different subroutines to deal
  // with the two categories of cases. The array
  // eqn_and_numerics_category flags which gas species belongs to
  // which category using the flags in this enum.
  // -----------------------------------------------------------------
  enum { NA, ANAL, IMPL };

  // process-specific configuration data (if any)
  struct Config {
    Config() {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
    Real dtsub_soa_fixed = -1;
    int ntot_soamode = 4;

    // Default Aging:
    // 0: Accumulation - No
    // 1: Aitken - No
    // 2: Coarse - No
    // 3: PrimaryCarbon - Yes
    bool l_mode_can_age[num_mode] = {false, false, false, true};

    bool calculate_gas_uptake_coefficient = true;

    // Do we have NH3? Not something supported at this time.
    static constexpr bool igas_nh3 = false;
  };

  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 gas/aersol exchange"; }

  // init -- initializes the implementation with MAM4's configuration
  void init(const AeroConfig &aero_config,
            const Config &process_config = Config()) {}

  // validate -- validates the given atmospheric state and prognostics against
  // assumptions made by this implementation, returning true if the states are
  // valid, false if not
  KOKKOS_INLINE_FUNCTION
  bool validate(const AeroConfig &config, const ThreadTeam &team,
                const Atmosphere &atm, const Surface &sfc,
                const Prognostics &progs) const {
    // Make sure relevant atmospheric quantities are physical.
    const int nk = atm.num_levels();
    int violations = 0;
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(team, nk),
        [&](int k, int &violation) {
          if ((atm.temperature(k) < 0) || (atm.pressure(k) < 0) ||
              (atm.vapor_mixing_ratio(k) < 0)) {
            violation = 1;
          }
        },
        violations);

    if (violations == 0) { // all clear so far
      // Check for negative mixing ratios.
      if (!progs.quantities_nonnegative(team)) {
        ++violations;
      }
    }
    return (violations > 0);
  }

  // compute_tendencies -- computes tendencies and updates diagnostics
  // NOTE: that both diags and tends are const below--this means their views
  // NOTE: are fixed, but the data in those views is allowed to vary.
  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atm,
                          const Surface &sfc, const Prognostics &progs,
                          const Diagnostics &diags,
                          const Tendencies &tends) const {}

private:
  // Gas-Aerosol-Exchange-specific configuration
  Config config_;
};

namespace gasaerexch {

KOKKOS_INLINE_FUNCTION
void mam_gasaerexch_1subarea_1gas_nonvolatile(
    const Real dt, const Real qgas_netprod_otrproc,
    const Real uptkaer[GasAerExch::num_mode], Real &qgas_cur, Real &qgas_avg,
    Real qaer_cur[GasAerExch::num_mode]) {

  // qgas_netprod_otrproc = gas net production rate from other processes
  //    such as gas-phase chemistry and emissions (mol/mol/s)
  // this allows the condensation (gasaerexch) routine to apply production and
  // condensation loss
  //    together, which is more accurate numerically
  // NOTE - must be >= zero, as numerical method can fail when it is negative
  // NOTE - currently only the values for h2so4 should be non-zero

  const int num_mode = GasAerExch::num_mode;

  Real qaer_prev[num_mode];
  // Save current values as the "previous" value for new time step
  for (int i = 0; i < num_mode; ++i)
    qaer_prev[i] = qaer_cur[i];
  // misc. quantities that will occur repeatedly later equations

  // the uptake coefficient of a single gas species
  // summed over all aerosol modes
  Real total_uptake_coeff_of_all_modes = 0;
  for (int i = 0; i < num_mode; ++i)
    total_uptake_coeff_of_all_modes += uptkaer[i];

  const Real tmp_kxt = total_uptake_coeff_of_all_modes * dt; // uptake rate * dt
  const Real tmp_pxt = qgas_netprod_otrproc * dt;            // prodc. rate * dt

  // zqgas_init(was tmp_q1) = mix - rat at t = tcur zqgas_init = qgas_prv
  const Real zqgas_init = qgas_cur;

  // zqgas_end (was tmp_q3) = mix-rat at t=tcur+dt
  // zqgas_avg (was tmp_q4) = avg mix-rat between t=tcur and t=tcur+dt
  if (tmp_kxt < 1.0e-20) {
    // Consider uptake to aerosols = 0.0, hence there is no change in aerosol
    // mass; gas concentration is updated using the passed-in production rate
    // which is considered step-wise constant
    qgas_cur = zqgas_init + tmp_pxt;
    qgas_avg = zqgas_init + tmp_pxt * 0.5;
  } else {
    // tmp_kxt >= 1.0e-20_wp: there is non-negligible condensation;
    // calculate amount of gas condensation and update gas concentration.
    Real zqgas_end;
    Real zqgas_avg;
    if (tmp_kxt > 0.001) {
      // Stronger condesation, no worry about division by zero or singularity;
      // calculate analytical solution assuming step-wise constant condensation
      // coeff.
      const Real zqgas_equil = tmp_pxt / tmp_kxt;
      zqgas_end =
          (zqgas_init - zqgas_equil) * mam4::exp(-tmp_kxt) + zqgas_equil;
      zqgas_avg =
          (zqgas_init - zqgas_equil) * (1.0 - mam4::exp(-tmp_kxt)) / tmp_kxt +
          zqgas_equil;
    } else {
      // Weaker condensation, use Taylor expansion to avoid large error
      // resulting from small denominator
      const Real tmp_kxt2 = tmp_kxt * tmp_kxt;
      zqgas_end = zqgas_init * (1.0 - tmp_kxt + tmp_kxt2 * 0.5) +
                  tmp_pxt * (1.0 - tmp_kxt * 0.5 + tmp_kxt2 / 6.0);
      zqgas_avg = zqgas_init * (1.0 - tmp_kxt * 0.5 + tmp_kxt2 / 6.0) +
                  tmp_pxt * (0.5 - tmp_kxt / 6.0 + tmp_kxt2 / 24.0);
    }
    qgas_cur = zqgas_end;
    qgas_avg = zqgas_avg;
    // amount of condensed mass
    const Real tmp_qdel_cond = (zqgas_init + tmp_pxt) - zqgas_end;

    // Distribute the condensed mass to different aerosol modes
    for (int n = 0; n < num_mode; ++n) {
      if (0 < uptkaer[n])
        qaer_cur[n] =
            qaer_prev[n] +
            tmp_qdel_cond * (uptkaer[n] / total_uptake_coeff_of_all_modes);
    }
  }
}

//------------------------------------------------------------------------
// gas_diffusivity       ! (m2/s)
KOKKOS_INLINE_FUNCTION
Real gas_diffusivity(
    const Real &T_in_K,   // temperature (K)
    const Real &p_in_atm, // pressure (atmospheres)
    const Real mw_gas,    // molec. weight of the condensing gas (g/mol)
    const Real vd_gas)    // molec. diffusion volume of the condensing gas
{

  constexpr Real onethird = 1.0 / 3.0;
  const Real dgas =
      (1.0e-3 * pow(T_in_K, 1.75) * mam4::sqrt(1. / mw_gas + 0.035)) /
      (p_in_atm * pow((pow(vd_gas, onethird) + 2.7189), 2.0));
  const Real gas_diffusivity = dgas * 1.0e-4;

  return gas_diffusivity;
}

//-----------------------------------------------------------------------
//  mean_molecular_speed    ! (m/s)
KOKKOS_INLINE_FUNCTION
Real mean_molecular_speed(
    const Real &temp,          // temperature (K)
    const Real rmw,            // molec. weight (g/mol)
    const Real r_universal_mJ, // universal gas constant (mJ/K mol)
    const Real pi) {
  // BAD CONSTANTS
  const Real mean_molecular_speed = 145.5 * mam4::sqrt(temp / rmw);

  return mean_molecular_speed;
}

//------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
Real fuchs_sutugin(const Real &D_p, const Real &gasfreepath,
                   const Real accomxp283, const Real accomxp75) {
  const Real knudsen = 2.0 * gasfreepath / D_p;

  // fkn = ( 0.75*accomcoef*(1. + xkn) ) /
  //       ( xkn*xhn + xkn + 0.283*xkn*accomcoef + 0.75*accomcoef )

  const Real fuchs_sutugin =
      (accomxp75 * (1.0 + knudsen)) /
      (knudsen * (knudsen + 1.0 + accomxp283) + accomxp75);

  return fuchs_sutugin;
}

KOKKOS_INLINE_FUNCTION
void gas_aer_uptkrates_1box1gas(const Real accom, const Real gasdiffus,
                                const Real gasfreepath, const Real beta_inp,
                                const Real dgncur_awet[GasAerExch::num_mode],
                                const Real lnsg[GasAerExch::num_mode],
                                Real uptkaer[GasAerExch::num_mode]) {
  //!                         /
  //!   computes   uptkrate = | dx  dN/dx  gas_conden_rate(Dp(x))
  //!                         /
  //!   using Gauss-Hermite quadrature of order nghq=2
  //!
  //!       Dp = particle diameter (cm)
  //!       x = ln(Dp)
  //!       dN/dx = log-normal particle number density distribution
  //!       gas_conden_rate(Dp) = 2 * pi * gasdiffus * Dp * F(Kn,ac)
  //!           F(Kn,ac) = Fuchs-Sutugin correction factor
  //!           Kn = Knudsen number
  //!           ac = accomodation coefficient

  const Real tworootpi = 2 * mam4::sqrt(Constants::pi);
  const Real root2 = mam4::sqrt(2.0);
  const Real one = 1.0;
  const Real two = 2.0;

  // Dick's old version
  // integer, parameter :: nghq = 2
  // real(wp), save :: xghq(nghq), wghq(nghq) ! quadrature abscissae and
  // weights data xghq / 0.70710678, -0.70710678 / data wghq / 0.88622693,
  // 0.88622693 /

  // NOTE: it looks like the refractored code is still using Dick's old version.

  //-----------------------------------------------------------------------

  constexpr int nghq = 2;
  const Real xghq[nghq] = {7.071067690849304E-01, -7.071067690849304E-01};
  const Real wghq[nghq] = {8.862269520759583E-01, 8.862269520759583E-01};

  const Real accomxp283 = accom * 0.283;
  const Real accomxp75 = accom * 0.75;

  // outermost loop over all modes
  for (int n = 0; n < GasAerExch::num_mode; ++n) {
    const Real lndpgn = mam4::log(dgncur_awet[n]); // (m)

    // beta = dln(uptake_rate)/dln(D_p)
    //      = 2.0 in free molecular regime, 1.0 in continuum regime
    // if uptake_rate ~= a * (D_p**beta), then the 2 point quadrature
    // is very accurate
    Real beta = 0;
    if (std::abs(beta_inp - 1.5) > 0.5) {
      // D_p = dgncur_awet(n) * mam4::exp( 1.5*(lnsg[n]**2) )
      const Real D_p = dgncur_awet[n];
      const Real knudsen = two * gasfreepath / D_p;

      // tmpa = dln(fuchs_sutugin)/d(knudsen)
      const Real tmpa =
          one / (one + knudsen) -
          (two * knudsen + one + accomxp283) /
              (knudsen * (knudsen + one + accomxp283) + accomxp75);
      beta = one - knudsen * tmpa;
      beta = mam4::max(one, mam4::min(two, beta));
    } else {
      beta = beta_inp;
    }
    const Real constant =
        tworootpi * mam4::exp(beta * lndpgn + 0.5 * pow(beta * lnsg[n], 2.0));

    // sum over gauss-hermite quadrature points
    Real sumghq = 0.0;
    for (int iq = 0; iq < nghq; ++iq) {
      const Real lndp =
          lndpgn + beta * lnsg[n] * lnsg[n] + root2 * lnsg[n] * xghq[iq];
      const Real D_p = mam4::exp(lndp);

      const Real hh = fuchs_sutugin(D_p, gasfreepath, accomxp283, accomxp75);
      sumghq += wghq[iq] * D_p * hh / pow(D_p, beta);
    }
    // gas-to-aerosol mass transfer rates
    uptkaer[n] = constant * gasdiffus * sumghq;
  }
} // gas_aer_uptkrates_1box1gas

} // namespace gasaerexch

KOKKOS_INLINE_FUNCTION
void mam_gasaerexch_1subarea(
    const int jtsubstep,                                            // in
    const Real dtsubstep,                                           // in
    const Real temp,                                                // in
    const Real pmid,                                                // in
    const Real aircon,                                              // in
    const int n_mode,                                               // in
    Real qgas_cur[gasaerexch::max_gas],                             // inout
    Real qgas_avg[gasaerexch::max_gas],                             // inout
    const Real qgas_netprod_otrproc[gasaerexch::max_gas],           // in
    Real qaer_cur[gasaerexch::max_aer][mam4::gasaerexch::max_mode], // inout
    Real qnum_cur[gasaerexch::max_mode],                            // inout
    const Real dgn_awet[gasaerexch::max_mode],                      // in
    Real uptkaer[gasaerexch::max_gas][mam4::gasaerexch::max_mode],  // inout
    Real &uptkrate_h2so4) {                                         // inout

  using mam4::exp;
  using mam4::log;
  using mam4::gasaerexch::max_aer;
  using mam4::gasaerexch::max_gas;
  using mam4::gasaerexch::max_mode;
  using mam4::gasaerexch::nsoa;

  constexpr int mode_aging_optaa[max_mode] = {0, 0, 0, 1, 0};

  Real qgas_prv[max_gas];
  Real qaer_prv[max_aer][max_mode];
  Real uptkrate[max_mode];
  constexpr int ntot_amode = AeroConfig::num_modes();
  // BAD CONSTANT
  Real alnsg_aer[max_mode] = {Real(mam4::log(1.8))};
  // sigmag_amode : assumed geometric standard deviation of particle size
  // distribution
  for (int imode = 0; imode < ntot_amode; ++imode) {
    const Real sigmag_amode = modes(imode).mean_std_dev;
    alnsg_aer[imode] = mam4::log(sigmag_amode);
  }

  // using c++ indexing (fortran index -1)
  constexpr int lmap_aer[max_aer][max_mode] = {
      {8, 15, 24, -1},  {6, 14, 21, -1},  {7, -1, 23, 27},  {9, -1, 22, 28},
      {11, 16, 20, -1}, {10, -1, 19, -1}, {12, 17, 25, 29},
  };

  const Real r_universal_mJ = 1000 * Constants::r_gas; // [mJ/(K mol)]
  const Real r_pi = Constants::pi;
  // BAD CONSTANT
  // SOAG, H2SO4
  const Real mw_gas[max_gas] = {1.500000000000000E+02, 9.807840000000000E+01};
  const Real vol_molar_gas[max_gas] = {6.563265306122449E+01,
                                       4.288000000000000E+01};
  const Real accom_coef_gas[max_gas] = {6.5000E-01, 6.5000E-01};

  // igas_h2so4,
  constexpr int igas_nh3 = -1;
  constexpr int igas_hno3 = -1;
  constexpr int igas_hcl = -1;
  // constexpr int igas_soa=0;
  constexpr int igas_h2so4 = 1;

  // constexpr int iaer_soa=0;
  constexpr int iaer_so4 = 1;
  constexpr int iaer_nh4 = -1;
  // constexpr int iaer_no3=-1;
  // constexpr int iaer_cl=-1;

  // Initialize qgas_avg
  for (int igas = 0; igas < max_gas; ++igas) {
    qgas_avg[igas] = 0.0;
  }

  // Calculate gas uptake (mass transfer) rates
  if (jtsubstep == 0) {
    // pressure (atmospheres)
    // BAD CONSTANT
    const Real p_in_atm = pmid / 1.013e5;
    for (int igas = 0; igas < max_gas; ++igas) {
      // gas_diffus[igas] = gas_diffusivity(temp, tmpa, mw_gas[igas],
      // vol_molar_gas[igas]);
      const Real gas_diffus_igas = mam4::gasaerexch::gas_diffusivity(
          temp, p_in_atm, mw_gas[igas], vol_molar_gas[igas]);

      // tmpb = mean_molecular_speed(temp, mw_gas[igas]);
      // gas mean free path (m)
      const Real molecular_speed = mam4::gasaerexch::mean_molecular_speed(
          temp, mw_gas[igas], r_universal_mJ, r_pi);

      const Real gas_freepath_igas = 3.0 * gas_diffus_igas / molecular_speed;

      mam4::gasaerexch::gas_aer_uptkrates_1box1gas(
          accom_coef_gas[igas], gas_diffus_igas, gas_freepath_igas, 0.0,
          dgn_awet, alnsg_aer, uptkrate);
      const int iaer = igas;
      for (int n = 0; n < ntot_amode; ++n) {
        if (lmap_aer[iaer][n] > 0 || mode_aging_optaa[n] > 0) {
          uptkaer[igas][n] = uptkrate[n] * (qnum_cur[n] * aircon);
        } else {
          uptkaer[igas][n] = 0.0;
        }
      }
    }

    for (int igas = 0; igas < max_gas; ++igas) {
      // use cam5.1.00 uptake rates
      if (igas < nsoa) {
        // BAD CONSTANT
        for (int imode = 0; imode < ntot_amode; ++imode) {
          uptkaer[igas][imode] = uptkaer[igas_h2so4][imode] * 0.81;
        }
        // imode
      } // igas <= nso
      if (igas == igas_nh3) {
        // BAD CONSTANT
        for (int imode = 0; imode < ntot_amode; ++imode) {
          uptkaer[igas][imode] = uptkaer[igas_h2so4][imode] * 2.08;
        } // imode
      }   // igas == igas_nh3
    }     // igas
    // uptkrate_h2so4 = 0;
    for (int n = 0; n < ntot_amode; ++n)
      uptkrate_h2so4 += uptkaer[igas_h2so4][n];
  }

  // Do SOA
  mam4::gasaerexch::mam_soaexch_1subarea(dtsubstep, temp, pmid,        // in
                                         qgas_cur, qgas_avg, qaer_cur, // inout
                                         uptkaer);                     // in

  // Do other gases (that are assumed non-volatile) with no time sub-stepping
  for (int igas = nsoa; igas < max_gas; ++igas) {
    const int iaer = igas;
    qgas_prv[igas] = qgas_cur[igas];
    for (int n = 0; n < n_mode; ++n) {
      qaer_prv[iaer][n] = qaer_cur[iaer][n];
    }
  }

  for (int igas = nsoa; igas < max_gas; ++igas) {
    const int iaer = igas;
    if (igas == igas_hno3 || igas == igas_hcl)
      continue;

    Real tmpa = 0.0;
    for (int imode = 0; imode < n_mode; ++imode) {
      tmpa += uptkaer[igas][imode];
    } // imode

    const Real tmp_kxt = tmpa * dtsubstep;
    const Real tmp_pxt = qgas_netprod_otrproc[igas] * dtsubstep;
    const Real tmp_q1 = qgas_prv[igas];

    // BAD CONSTANT
    if (tmp_kxt >= 1.0e-20) {
      Real tmp_q3 = 0.0;
      Real tmp_q4 = 0.0;
      if (tmp_kxt > 0.001) {
        const Real tmp_pok = tmp_pxt / tmp_kxt;
        tmp_q3 = (tmp_q1 - tmp_pok) * mam4::exp(-tmp_kxt) + tmp_pok;
        tmp_q4 = (tmp_q1 - tmp_pok) * (1.0 - mam4::exp(-tmp_kxt)) / tmp_kxt +
                 tmp_pok;
      } else {
        const Real tmp_kxt2 = tmp_kxt * tmp_kxt;
        tmp_q3 = tmp_q1 * (1.0 - tmp_kxt + tmp_kxt2 * 0.5) +
                 tmp_pxt * (1.0 - tmp_kxt * 0.5 + tmp_kxt2 / 6.0);
        tmp_q4 = tmp_q1 * (1.0 - tmp_kxt * 0.5 + tmp_kxt2 / 6.0) +
                 tmp_pxt * (0.5 - tmp_kxt / 6.0 + tmp_kxt2 / 24.0);
      }
      qgas_cur[igas] = tmp_q3;
      const Real tmp_qdel_cond = (tmp_q1 + tmp_pxt) - tmp_q3;
      qgas_avg[igas] = tmp_q4;
      for (int n = 0; n < n_mode; ++n) {
        if (uptkaer[igas][n] <= 0.0)
          continue;
        const Real tmpc = tmp_qdel_cond * (uptkaer[igas][n] / tmpa);
        qaer_cur[iaer][n] = qaer_prv[iaer][n] + tmpc;
      }
    } else {
      qgas_cur[igas] = tmp_q1 + tmp_pxt;
      qgas_avg[igas] = tmp_q1 + tmp_pxt * 0.5;
    }
  }

  if (igas_nh3 > 0) {
    // Do not allow nh4 to exceed 2*so4 (molar basis)
    const int iaer = iaer_nh4;
    const int igas = igas_nh3;
    for (int n = 0; n < n_mode; ++n) {
      if (uptkaer[igas][n] <= 0.0)
        continue;
      const Real tmpa = qaer_cur[iaer][n] - 2.0 * qaer_cur[iaer_so4][n];
      if (tmpa > 0.0) {
        qaer_cur[iaer][n] = qaer_cur[iaer][n] - tmpa;
        qgas_cur[igas] = qgas_cur[igas] + tmpa;
        qgas_avg[igas] = qgas_avg[igas] + tmpa * 0.5;
      }
    }
  }
}

} // namespace mam4

#endif
