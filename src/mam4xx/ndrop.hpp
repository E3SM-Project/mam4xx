#ifndef MAM4XX_NDROP_HPP
#define MAM4XX_NDROP_HPP

#include "aero_config.hpp"
#include "aero_model.hpp"
#include "conversions.hpp"
#include "mam4_math.hpp"
#include "mam4_types.hpp"
#include "utils.hpp"
#include "wv_sat_methods.hpp"

#include <ekat_math_utils.hpp>

namespace mam4 {

namespace ndrop {

using View1D = DeviceType::view_1d<Real>;
using View2D = DeviceType::view_2d<Real>;
using View3D = DeviceType::view_3d<Real>;
using ConstView2D = DeviceType::view_2d<const Real>;

// number of vertical levels
constexpr int pver = mam4::nlev;
// Top level for troposphere cloud physics
constexpr int psat = 6; //  number of supersaturations to calc ccn concentration
constexpr int maxd_aspectype = 14;
// BAD CONSTANT
// reference temperature [K] (from mam4)
// approximation of sea level h2o freezing point
constexpr Real t0 = 273.0;
// reference pressure [Pa] (from mam4)
// pressure at sea level [Pa] (1 atm)
constexpr Real p0 = 1013.25e2;
// BAD CONSTANT
// surface tension of water w/respect to air (N/m)
constexpr Real surften = 0.076;
// total number of mode number conc + mode species
constexpr int ncnst_tot = 25;
// max number of species in a mode
constexpr int nspec_max = 8;

KOKKOS_INLINE_FUNCTION
void get_aer_mmr_sum(const int imode, const int nspec,
                     const Real state_q[aero_model::pcnst],
                     const Real qcldbrn1d[maxd_aspectype],
                     Real &vaerosolsum_icol, Real &hygrosum_icol) {

  // @param[in] imode  mode index
  // @param[in] nspec  total # of species in mode imode
  // @param[in] state_q interstitial aerosol mass mixing ratios [kg/kg]
  // @param[in] qcldbrn1d cloud-borne aerosol mass mixing ratios [kg/kg]

  const Real zero = 0;
  // sum to find volume conc [m3/kg]
  vaerosolsum_icol = zero;
  // sum to bulk hygroscopicity of mode [m3/kg]
  hygrosum_icol = zero;

  // Start to compute bulk volume conc / hygroscopicity by summing over species
  // per mode.
  for (int lspec = 0; lspec < nspec; ++lspec) {
    // Fortran indexing to C++
    const int type_idx = AeroConfig::lspectype_amode(lspec, imode);
    // density at species / mode indices [kg/m3]
    const Real density_sp =
        AeroConfig::specdens_amode(type_idx); // species density
    // hygroscopicity at species / mode indices [dimensionless]
    const Real hygro_sp =
        AeroConfig::spechygro(type_idx); // species hygroscopicity
    // Fortran indexing to C++
    // index of species in state_q array
    const int spc_idx = AeroConfig::lmassptr_amode(lspec, imode);
    // aerosol volume mixing ratio [m3/kg]
    const Real vol = mam4::max(state_q[spc_idx] + qcldbrn1d[lspec], zero) /
                     density_sp; // volume = mmr/density
    vaerosolsum_icol += vol;
    hygrosum_icol += vol * hygro_sp; // bulk hygroscopicity
  }                                  // end
} // end get_aer_mmr_sum

KOKKOS_INLINE_FUNCTION
void get_aer_num(const Real voltonumbhi_amode, const Real voltonumblo_amode,
                 const int num_idx, const Real state_q[aero_model::pcnst],
                 const Real air_density, const Real vaerosol,
                 const Real qcldbrn1d_num, Real &naerosol) {

  // input arguments
  // @param[in] imode        mode index
  // @param[in] state_q(:,:) interstitial aerosol number mixing ratios [#/kg]
  // @param[in] air_density        air density [kg/m3]
  // @param[in] vaerosol(:)  volume conc [m3/m3]
  // @param[in] qcldbrn1d_num(:) cloud-borne aerosol number mixing ratios
  // [#/kg]
  // @param[out] naerosol(:)  number conc [#/m3]

  // convert number mixing ratios to number concentrations
  // Use bulk volume conc found previously to bound value
  // or contains both species and modes concentrations ?
  naerosol = (state_q[num_idx] + qcldbrn1d_num) * air_density;

  // adjust number so that dgnumlo < dgnum < dgnumhi
  naerosol = utils::min_max_bound(vaerosol * voltonumbhi_amode,
                                  vaerosol * voltonumblo_amode, naerosol);

} // end get_aer_num

KOKKOS_INLINE_FUNCTION
void maxsat(
    // TODO: what are zeta and eta?
    const Real zeta,                         // [dimensionless]
    const Real eta[AeroConfig::num_modes()], // [dimensionless]
    const Real nmode,                        // number of modes
    // critical supersaturation for number mode radius [fraction]
    const Real smode_crit[AeroConfig::num_modes()],
    Real &smax // maximum supersaturation [fraction] (output)
) {

  /* calculates maximum supersaturation for multiple
        competing aerosol modes.

    Abdul-Razzak and Ghan, A parameterization of aerosol activation.
    2. Multiple aerosol types. J. Geophys. Res., 105, 6837-6844. */

  // abdul-razzak functions of width
  Real ar_func1[AeroConfig::num_modes()];
  Real ar_func2[AeroConfig::num_modes()];

  Real const small = 1e-20;     /*BAD CONSTANT*/
  Real const mid = 1e5;         /*BAD CONSTANT*/
  Real const big = 1.0 / small; /*BAD CONSTANT*/
  Real sum = 0;
  Real g1, g2;
  bool weak_forcing = true; // whether forcing is sufficiently weak or not

  for (int m = 0; m < nmode; m++) {
    if (zeta > mid * eta[m] || smode_crit[m] * smode_crit[m] > mid * eta[m]) {
      // weak forcing. essentially none activated
      smax = small;
    } else {
      // significant activation of this mode. calc activation of all modes.
      weak_forcing = false;
      break;
    }
  }

  // if the forcing is weak, return
  if (weak_forcing)
    return;

  for (int m = 0; m < nmode; m++) {
    ar_func1[m] =
        0.5 * mam4::exp(2.5 * square(mam4::log(modes(m).mean_std_dev)));
    ar_func2[m] = 1.0 + 0.25 * mam4::log(modes(m).mean_std_dev);
    if (eta[m] > small) {
      g1 = (zeta / eta[m]) * mam4::sqrt(zeta / eta[m]);
      g2 = (smode_crit[m] / mam4::sqrt(eta[m] + 3.0 * zeta)) *
           mam4::sqrt(smode_crit[m] / mam4::sqrt(eta[m] + 3.0 * zeta));
      sum += (ar_func1[m] * g1 + ar_func2[m] * g2) /
             (smode_crit[m] * smode_crit[m]);
    } else {
      sum = big;
    }
  }
  smax = 1.0 / mam4::sqrt(sum);
  return;
} // end maxsat

KOKKOS_INLINE_FUNCTION
void loadaer(const Real state_q[aero_model::pcnst], Real air_density,
             const int phase,
             const Real voltonumbhi_amode[AeroConfig::num_modes()],
             const Real voltonumblo_amode[AeroConfig::num_modes()],
             const Real qcldbrn1d[maxd_aspectype][AeroConfig::num_modes()],
             const Real qcldbrn1d_num[AeroConfig::num_modes()],
             Real naerosol[AeroConfig::num_modes()],
             Real vaerosol[AeroConfig::num_modes()],
             Real hygro[AeroConfig::num_modes()]) {

  // return aerosol number, volume concentrations, and bulk hygroscopicity at
  // one specific column and level

  // input arguments
  // @param [in] state_q(:)         aerosol mmrs [kg/kg]
  // @param [in] air_density        air density [kg/m3]
  // @param [in]  phase     phase of aerosol: 1 for interstitial, 2 for
  //                                          cloud-borne, 3 for sum

  // output arguments
  // @param [out] naerosol(ntot_amode) number conc [#/m3]
  // @param [out] vaerosol(ntot_amode) volume conc [m3/m3]
  // @param [out] hygro(ntot_amode)    bulk hygroscopicity of mode [-]

  // optional input arguments; we assumed that this input is not optional
  // cloud-borne aerosol mass or number mixing ratios [kg/kg or #/kg]
  // @param [in] qcldbrn1d(:,:), qcldbrn1d_num(:)

  // Currently supports only phase 1 (interstitial) and 3 (interstitial+cldbrn)
  // but... what's phase 2???
  if (phase != 1 && phase != 3) {
    Kokkos::abort("phase error in loadaer");
  }

  const Real nmodes = AeroConfig::num_modes();
  // BAD CONSTANT
  const Real even_smaller_val = 1.0e-30;

  // assume it's present, so we don't need to do this
  // qcldbrn_local(:,:nspec) = qcldbrn1d(:,:nspec)

  // Sum over all species within imode to get bulk hygroscopicity and volume
  // conc phase == 1 is interstitial only. phase == 3 is interstitial + cldborne
  // Assumes iphase =1 or 3, so interstitial is always summed, added with cldbrn
  // when present iphase = 2 would require alternate logic from following
  // subroutine

  const Real zero = 0;
  Real qcldbrn1d_imode[AeroConfig::num_aerosol_ids()] = {};
  for (int imode = 0; imode < nmodes; ++imode) {
    Real vaerosolsum = zero;
    Real hygrosum = zero;
    const Real nspec = AeroConfig::nspec_amode(imode);

    for (int ispec = 0; ispec < nspec; ++ispec) {
      qcldbrn1d_imode[ispec] = qcldbrn1d[ispec][imode];
    }

    get_aer_mmr_sum(imode, nspec, state_q, qcldbrn1d_imode, vaerosolsum,
                    hygrosum);

    //  Finalize computation of bulk hygroscopicity and volume conc
    // NOTE: maybe use safe_denominator()?
    if (vaerosolsum > even_smaller_val) {
      hygro[imode] = hygrosum / vaerosolsum;
      vaerosol[imode] = vaerosolsum * air_density;
    } else {
      hygro[imode] = zero;
      vaerosol[imode] = zero;
    } //

    // Compute aerosol number concentration
    // Fortran indexing to C++
    const int num_idx = AeroConfig::numptr_amode(imode);
    get_aer_num(voltonumbhi_amode[imode], voltonumblo_amode[imode], num_idx,
                state_q, air_density, vaerosol[imode], qcldbrn1d_num[imode],
                naerosol[imode]);

  } // end imode
} // loadaer

KOKKOS_INLINE_FUNCTION
void ccncalc(const Real state_q[aero_model::pcnst], const Real tair,
             const Real qcldbrn[maxd_aspectype][AeroConfig::num_modes()],
             const Real qcldbrn_num[AeroConfig::num_modes()],
             const Real air_density,
             const Real voltonumbhi_amode[AeroConfig::num_modes()],
             const Real voltonumblo_amode[AeroConfig::num_modes()],
             const Real exp45logsig[AeroConfig::num_modes()],
             const Real alogsig[AeroConfig::num_modes()], Real ccn[psat]) {

  // calculates number concentration of aerosols activated as CCN at
  // supersaturation supersat.
  // assumes an internal mixture of multiple externally-mixed aerosol modes
  // NOTE: CGS units

  // Ghan et al., Atmos. Res., 1993, 198-221.

  // input arguments
  // @param [in] state_q(:,:,:) aerosol mmrs [kg/kg]
  // @param [in] tair(:,:)     air temperature [K]
  // @param [in] qcldbrn(:,:,:,:), qcldbrn_num(:,:,:) cloud-borne aerosol mass
  // / number  mixing ratios [kg/kg or #/kg]
  // @param [in] air_density(pcols,pver)       air density [kg/m3]

  // output arguments
  // ccn(pcols,pver,psat) number conc of aerosols activated at supersat [#/m3]
  // qcldbrn  1) icol 2) ispec 3) kk 4)imode
  // state_q 1) icol 2) kk 3) imode and ispec
  const Real zero = 0;
  // BAD CONSTANT
  const Real nconc_thresh = 1.0e-3;
  const Real twothird = 2.0 / 3.0;
  const Real two = 2.0;
  const Real three_fourths = 3.0 / 4.0;
  const Real half = 0.5;
  const Real one = 1;

  const Real per_m3_to_per_cm3 = 1.0e-6;
  //// const Real percent_to_fraction = 0.01;
  // super(:)=supersat(:)*percent_to_fraction
  // supersaturation [fraction]
  const Real sq2 = mam4::sqrt(2.0);
  const Real two_over_root27 = 2.0 / mam4::sqrt(27.0);
  // BAD CONSTANT
  // supersaturation (%) to determine ccn concentration
  const Real super[psat] = {0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01};
  // phase of aerosol
  const int phase = 3; // interstitial + cloudborne
  const int nmodes = AeroConfig::num_modes();

  const Real mwh2o = Constants::molec_weight_h2o * 1e3; // [kg/kmol]
  const Real r_universal = Constants::r_gas * 1e3;      // [J/K/kmol]
  const Real rhoh2o = Constants::density_h2o;
  const Real pi = Constants::pi;

  // surface tension coefficient [m-K] (from mam4)
  // surface tension has SI units [N m^-1] and [dyn cm^-1] in CGS, with
  // alternative definitions being [J m^-2] in SI and [erg cm^-2] in CGS
  // maybe it indicates MKS units?
  const Real surften_coef = two * mwh2o * surften / (r_universal * rhoh2o);

  // surface tension parameter [m]
  const Real aparam = surften_coef / tair;
  // [m^(3/2)]
  const Real ssat_coeff = two_over_root27 * aparam * mam4::sqrt(aparam);

  // interstitial + activated aerosol number conc [#/m3]
  Real naerosol[AeroConfig::num_modes()] = {zero};
  // interstitial + activated aerosol volume conc [m3/m3]
  Real vaerosol[AeroConfig::num_modes()] = {zero};
  Real hygro[AeroConfig::num_modes()] = {zero};

  loadaer(state_q, air_density, phase, voltonumbhi_amode, voltonumblo_amode,
          qcldbrn, qcldbrn_num, naerosol, vaerosol, hygro);

  for (int lsat = 0; lsat < psat; ++lsat) {
    ccn[lsat] = {zero};
  }
  for (int imode = 0; imode < nmodes; ++imode) {
    // critical supersaturation at mode radius [fraction]
    // here we assume "value shouldn't matter much since naerosol is small"
    // ^(from mam4)
    // NOTE: excuse me... what??
    Real ss_crit_imode = 1;
    if (naerosol[imode] > nconc_thresh) {
      // [dimensionless]
      const Real amcubecoef_imode = three_fourths / (pi * exp45logsig[imode]);
      // [m3]
      const Real amcube = amcubecoef_imode * vaerosol[imode] / naerosol[imode];
      // critical supersaturation at mode radius [unitless]
      ss_crit_imode = ssat_coeff / mam4::sqrt(hygro[imode] * amcube);
    }

    const Real argfactor_imode = twothird / (sq2 * alogsig[imode]);
    for (int lsat = 0; lsat < psat; ++lsat) {
      // [dimensionless]
      const Real arg_erf_ccn =
          argfactor_imode * mam4::log(ss_crit_imode / super[lsat]);
      ccn[lsat] += naerosol[imode] * half * (one - mam4::erf(arg_erf_ccn));
    }

  } // imode end

  for (int lsat = 0; lsat < psat; ++lsat) {
    ccn[lsat] *= per_m3_to_per_cm3; // convert from #/m3 to #/cm3
  }                                 // lsat

} /// ccncalc

KOKKOS_INLINE_FUNCTION
void qsat(const Real t, const Real p, Real &svp, Real &qsat) {
  //  ------------------------------------------------------------------
  // Purpose:
  //   Look up and return saturation vapor pressure from precomputed
  //   table, then calculate and return saturation specific humidity.
  //   Optionally return various temperature derivatives or enthalpy
  //   at saturation.
  // ------------------------------------------------------------------
  // Inputs
  // @param [in] t    Temperature
  // @param [in] p    Pressure
  // Outputs
  // @param [out] svp  Saturation vapor pressure
  // @param [out] qsat  Saturation specific humidity

  // Note. Fortran code uses a table lookup. In C++ version, we compute directly
  // from the function.
  svp = wv_sat_methods::wv_sat_svp_trans(t);
  qsat = wv_sat_methods::wv_sat_svp_to_qsat(svp, p);
  // Ensures returned svp is consistent with limiters on qsat.
  svp = mam4::min(svp, p);
} // qsat

KOKKOS_INLINE_FUNCTION
void ndrop_init(Real exp45logsig[AeroConfig::num_modes()],
                Real alogsig[AeroConfig::num_modes()], Real &aten,
                Real num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
                Real num2vol_ratio_max_nmodes[AeroConfig::num_modes()]) {
  const Real one = 1.0;
  const Real two = 2.0;
  const Real one_thousand = 1e3;
  for (int imode = 0; imode < AeroConfig::num_modes(); ++imode) {
    alogsig[imode] = mam4::log(modes(imode).mean_std_dev);
    exp45logsig[imode] = mam4::exp(4.5 * alogsig[imode] * alogsig[imode]);

    // voltonumbhi_amode
    num2vol_ratio_min_nmodes[imode] =
        one / conversions::mean_particle_volume_from_diameter(
                  modes(imode).max_diameter, modes(imode).mean_std_dev);
    // voltonumblo_amode
    num2vol_ratio_max_nmodes[imode] =
        one / conversions::mean_particle_volume_from_diameter(
                  modes(imode).min_diameter, modes(imode).mean_std_dev);

  } // imode

  // density of fresh water [kg/m^3]
  const Real rhoh2o = Constants::density_h2o;
  // [J/K/kmol]
  const Real r_universal = Constants::r_gas * one_thousand;
  // [kg/kmol]
  const Real mwh2o = Constants::molec_weight_h2o * one_thousand;
  // BAD CONSTANT
  // [m]
  aten = two * mwh2o * surften / (r_universal * t0 * rhoh2o);

} // end ndrop_init

KOKKOS_INLINE_FUNCTION
void activate_modal(const Real w_in, const Real wmaxf, const Real tair,
                    const Real rhoair, Real na[AeroConfig::num_modes()],
                    const Real volume[AeroConfig::num_modes()],
                    const Real hygro[AeroConfig::num_modes()],
                    const Real exp45logsig[AeroConfig::num_modes()],
                    const Real alogsig[AeroConfig::num_modes()],
                    const Real aten, Real fn[AeroConfig::num_modes()],
                    Real fm[AeroConfig::num_modes()],
                    Real fluxn[AeroConfig::num_modes()],
                    Real fluxm[AeroConfig::num_modes()], Real &flux_fullact) {
  //    ---------------------------------------------------------------------------------
  // Calculates number, surface, and mass fraction of aerosols activated as CCN
  // calculates flux of cloud droplets, surface area, and aerosol mass into
  // cloud assumes an internal mixture within each of up to nmode multiple
  // aerosol modes a gaussian spectrum of updrafts can be treated.
  //
  // Units: SI (MKS)
  //
  // Reference: Abdul-Razzak and Ghan, A parameterization of aerosol
  // activation.      2. Multiple aerosol types. J. Geophys. Res., 105,
  // 6837-6844.
  // ---------------------------------------------------------------------------------

  // input
  // @param [in] w_in      vertical velocity [m/s]
  // @param [in] wmaxf     maximum updraft velocity for integration [m/s]
  // @param [in] tair      air temperature [K]
  // @param [in] rhoair    air density [kg/m3]
  // @param [in] na(:)     aerosol number concentration [#/m3]
  // @param [in] nmode     number of aerosol modes
  // @param [in] volume(:) aerosol volume concentration [m3/m3]
  // @param [in] hygro(:)  hygroscopicity of aerosol mode [dimensionless]

  // output
  // @param [out] fn(:)        number fraction of aerosols activated
  // [fraction]
  // @param [out] fm(:)        mass fraction of aerosols activated [fraction]
  // @param [out] fluxn(:)     flux of activated aerosol number fraction into
  // cloud [m/s]
  // @param [out] fluxm(:)     flux of activated aerosol mass fraction into
  // cloud [m/s]
  // @param [out] flux_fullact flux of activated aerosol fraction assuming
  // 100% activation [m/s]

  // Note: We did not include  the optional parameters smax_prescribed(
  // supersaturation for secondary activation [fraction])

  // ---------------------------------------------------------------------------------
  // flux_fullact is used for consistency check -- this should match
  // (eddy_diff(k)*zs(k)) also, fluxm/flux_fullact gives fraction of aerosol
  // mass flux that is activated
  // ---------------------------------------------------------------------------------
  const Real zero = 0;
  const Real one = 1;
  const Real sq2 = mam4::sqrt(2.0);
  const Real two = 2;
  const Real three_fourths = 3.0 / 4.0;
  const Real twothird = 2.0 / 3.0;
  const Real half = 0.5;
  const Real small = 1.0e-39;

  constexpr int nmode = AeroConfig::num_modes();
  // BAD CONSTANT
  // return if aerosol number is negligible in the accumulation mode
  const int accum_idx = int(ModeIndex::Accumulation);
  if (na[accum_idx] < 1.0e-20) {
    return;
  }

  // return if vertical velocity is 0 or negative
  if (w_in <= zero) {
    return;
  }

  const Real rair = Constants::r_gas_dry_air;
  const Real rh2o = Constants::r_gas_h2o_vapor;
  // latent heat of evaporation [J/kg]
  const Real latvap = Constants::latent_heat_evap;
  // specific heat of dry air [J/kg/K]
  const Real cpair = Constants::cp_dry_air;
  // acceleration of gravity [m/s^2]
  const Real gravity = Constants::gravity;
  // density of fresh water [kg/m^3]
  const Real rhoh2o = Constants::density_h2o;
  const Real pi = Constants::pi;

  const Real pres = rair * rhoair * tair; // pressure [Pa]
  // Obtain Saturation vapor pressure (svp) and saturation specific humidity
  // (qs)
  //  water vapor saturation specific humidity [kg/kg]
  Real qs = zero;
  // saturation vapor pressure [Pa]
  Real svp = zero;
  qsat(tair, pres, svp, qs); // svp and qs are the outputs
  // change in qs with temperature  [(kg/kg)/T]
  const Real dqsdt = latvap / (rh2o * tair * tair) * qs;
  // [/m]
  const Real alpha =
      gravity * (latvap / (cpair * rh2o * tair * tair) - one / (rair * tair));
  // [m3/kg]
  const Real gamma = (one + latvap / cpair * dqsdt) / (rhoair * qs);
  // [s^(3/2)]
  // BAD CONSTANT
  // this should make eta big if na is very small.
  const Real etafactor2max = 1.0e10 / mam4::pow((alpha * wmaxf), 1.5);
  // vapor diffusivity [m2/s]
  const Real diff0 = 0.211e-4 * (p0 / pres) * mam4::pow(tair / t0, 1.94);
  // thermal conductivity [J / (m-s-K)]--converted to [J/m/s/deg]
  const Real conduct0 = (5.69 + 0.017 * (tair - t0)) * 4.186e2 * 1.0e-5;
  // thermodynamic function [m2/s]
  // gthermfac is same for all modes
  const Real gthermfac = one / (rhoh2o / (diff0 * rhoair * qs) +
                                latvap * rhoh2o / (conduct0 * tair) *
                                    (latvap / (rh2o * tair) - one));
  const Real beta = two * pi * rhoh2o * gthermfac * gamma; // [m2/s]
  const Real wnuc = w_in;
  const Real alw = alpha * wnuc;                 // [/s]
  const Real etafactor1 = alw * mam4::sqrt(alw); // [/ s^(3/2)]
  // [unitless]
  const Real zeta = twothird * mam4::sqrt(alw) * aten / mam4::sqrt(gthermfac);

  Real amcube[nmode] = {}; // cube of dry mode radius [m3]

  Real etafactor2[nmode] = {};
  Real lnsm[nmode] = {};

  // critical supersaturation for number mode radius [fraction]
  Real ssat_crit_imode[nmode] = {};

  Real eta[nmode] = {};
  // Here compute ssat_crit_imode, eta for all modes for maxsat calculation
  for (int imode = 0; imode < nmode; ++imode) {
    // BAD CONSTANT

    if (volume[imode] > small && na[imode] > small) {
      // number mode radius [m]
      // only if variable size dist
      amcube[imode] =
          three_fourths * volume[imode] / (pi * exp45logsig[imode] * na[imode]);
      // Growth coefficient Abdul-Razzak & Ghan 1998 eqn 16
      // should depend on mean radius of mode to account for gas kinetic
      // effects see Fountoukis and Nenes, JGR2005 and Meskhidze et al.,
      // JGR2006 for appropriate size to use for effective diffusivity.
      etafactor2[imode] = one / (na[imode] * beta * mam4::sqrt(gthermfac));
      // BAD CONSTANT
      if (hygro[imode] > 1.0e-10) {
        ssat_crit_imode[imode] =
            two * aten *
            // only if variable size dist
            mam4::sqrt(aten / (27.0 * hygro[imode] * amcube[imode]));
      } else {
        // BAD CONSTANT
        ssat_crit_imode[imode] = 100.0;
      } // hygro
    } else {
      ssat_crit_imode[imode] = one;
      // this should make eta big if na is very small.
      etafactor2[imode] = etafactor2max;
    } // volume
    // only if variable size dist
    lnsm[imode] = mam4::log(ssat_crit_imode[imode]);
    eta[imode] = etafactor1 * etafactor2[imode];
  } // end imode

  Real supersat = zero;

  maxsat(zeta, eta, nmode, ssat_crit_imode, supersat);
  // ([fraction]))
  const Real lnsupersat = mam4::log(supersat);

  // Use maximum supersaturation to calculate aerosol activation output
  for (int imode = 0; imode < nmode; ++imode) {
    // [unitless]
    const Real arg_erf_n =
        twothird * (lnsm[imode] - lnsupersat) / (sq2 * alogsig[imode]);

    fn[imode] = half * (one - mam4::erf(arg_erf_n)); // activated number

    const Real arg_erf_m = arg_erf_n - 1.5 * sq2 * alogsig[imode];
    fm[imode] = half * (one - mam4::erf(arg_erf_m)); // activated mass
    fluxn[imode] = fn[imode] * w_in; // activated aerosol number flux
    fluxm[imode] = fm[imode] * w_in; // activated aerosol mass flux
  }
  // is vertical velocity equal to flux of activated aerosol fraction assuming
  // 100% activation [m/s]?
  flux_fullact = w_in;

} // activate_modal

KOKKOS_INLINE_FUNCTION
void get_activate_frac(const Real state_q_kload[aero_model::pcnst],
                       const Real air_density_kload, const Real air_density_kk,
                       const Real wtke,
                       const Real tair, // in
                       const Real voltonumbhi_amode[AeroConfig::num_modes()],
                       const Real voltonumblo_amode[AeroConfig::num_modes()],
                       const Real exp45logsig[AeroConfig::num_modes()],
                       const Real alogsig[AeroConfig::num_modes()],
                       const Real aten, Real fn[AeroConfig::num_modes()],
                       Real fm[AeroConfig::num_modes()],
                       Real fluxn[AeroConfig::num_modes()],
                       Real fluxm[AeroConfig::num_modes()],
                       Real &flux_fullact) {

  // input arguments
  //  @param [in] state_q_kload(:)         aerosol mmrs at level from which to
  //  load aerosol [kg/kg]
  //  @param [in] cs_kload     air density at level from which to load aerosol
  //  [kg/m3]
  //  @param [in] cs_kk        air density at actual vertical level [kg/m3]
  //  @param [in] wtke        subgrid vertical velocity [m/s]
  //  @param [in]  tair        air temperature [K]

  // output arguments
  // @param [out]  fn(:)        number fraction of aerosols activated
  // [fraction]
  // @param [out]   fm(:)        mass fraction of aerosols activated
  // [fraction]
  // @param [out]   fluxn(:)     flux of activated aerosol number fraction
  // into cloud [m/s]
  // @param [out]   fluxm(:)     flux of activated aerosol mass fraction into
  // cloud [m/s]
  // @param [out]   flux_fullact flux of activated aerosol fraction assuming
  // 100% activation [m/s]
  const Real zero = 0;
  constexpr int nmodes = AeroConfig::num_modes();
  // NOTE: this gets set here, only to get passed to loadaer, where it's used
  // once for if(phase != 1 && phase != 3)
  const int phase = 1; // interstitial

  // NOTE: would it be better to initialize all of these to zero earlier?
  const Real qcldbrn[maxd_aspectype][AeroConfig::num_modes()] = {{zero}};
  const Real qcldbrn_num[AeroConfig::num_modes()] = {zero};

  Real naermod[nmodes] = {zero};  // aerosol number concentration [#/m^3]
  Real vaerosol[nmodes] = {zero}; // aerosol volume conc [m^3/m^3]
  Real hygro[nmodes] = {zero}; // hygroscopicity of aerosol mode [dimensionless]

  // load aerosol properties, assuming external mixtures
  loadaer(state_q_kload, air_density_kload, phase, voltonumbhi_amode,
          voltonumblo_amode, qcldbrn, qcldbrn_num, naermod, vaerosol, hygro);

  // BAD CONSTANT
  const Real wmax = 10.0;
  activate_modal(wtke, wmax, tair, air_density_kk, //   in
                 naermod, vaerosol, hygro,         //  in
                 exp45logsig, alogsig, aten, fn, fm, fluxn, fluxm,
                 flux_fullact); // out

} // get_activate_frac

template <class ViewType>
KOKKOS_INLINE_FUNCTION void update_from_cldn_profile(
    const Real cldn_col_in, const Real cldn_col_in_kp1, const Real dtinv,
    const Real wtke_col_in, const Real zs,
    const Real dz, // in
    const Real temp_col_in, const Real air_density, const Real air_density_kp1,
    const Real csbot_cscen,
    const Real state_q_col_in_kp1[aero_model::pcnst], // in
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const Real exp45logsig[AeroConfig::num_modes()],
    const Real alogsig[AeroConfig::num_modes()], const Real aten,
    ViewType raercol_nsav, ViewType raercol_nsav_kp1, ViewType raercol_cw_nsav,
    Real &nsource_col, // inout
    Real &qcld, Real factnum_col[AeroConfig::num_modes()],
    Real &eddy_diff, // out
    Real nact[AeroConfig::num_modes()], Real mact[AeroConfig::num_modes()]) {
  // clang-format off
  // input arguments
  // cldn_col_in(:)       cloud fraction [fraction] at kk
  // cldn_col_in_kp1(:)   cloud fraction [fraction] at min0(kk+1, pver);
  // dtinv                inverse time step for microphysics [s^{-1}]
  // wtke_col_in(:)       subgrid vertical velocity [m/s] at kk
  // zs(:)                inverse of distance between levels [m^-1]
  // dz(:)                geometric thickness of layers [m]
  // temp_col_in(:)       temperature [K]
  // air_density(:)       air density [kg/m^3] at kk
  // air_density_kp1      air density [kg/m^3] at min0(kk+1, pver);
  // csbot_cscen(:)       inverse normalized air density csbot(i)/cs(i,k)
  //                                       [dimensionless] [dimensionless]
  // state_q_col_in(:,:)    aerosol mmrs/ [kg/kg]

  // output arguments
  // raercol_nsav(:,:)    single column of saved aerosol mass, number mixing
  //                      ratios [#/kg or kg/kg]
  // raercol_cw_nsav(:,:) same as raercol but for cloud-borne phase [#/kg or kg/kg]
  // nsource_col(:)       droplet number mixing ratio source tendency [#/kg/s]
  // qcld(:)              cloud droplet number mixing ratio [#/kg]
  // factnum_col(:,:)     activation fraction for aerosol number [fraction]
  // eddy_diff(:)         diffusivity for droplets [m^2/s]
  // nact(:,:)            fractional aero. number activation rate [/s]
  // mact(:,:)            fractional aero. mass activation rate [/s]

  // clang-format on
  // ......................................................................
  // start of k-loop for calc of old cloud activation tendencies ..........
  //
  // rce-comment
  // changed this part of code to use current cloud fraction (cldn)
  // exclusively    consider case of cldo(:)=0, cldn(k)=1, cldn(k+1)=0
  // previous code (which used cldo below here) would have no cloud-base
  // activation       into layer k.  however, activated particles in k mix out
  // to k+1,       so they are incorrectly depleted with no replacement
  // BAD CONSTANT
  const Real cld_thresh = 0.01; //    threshold cloud fraction [fraction]
  // kp1 = min0(kk+1, pver);
  constexpr int ntot_amode = AeroConfig::num_modes();
  const Real zero = 0;
  if (cldn_col_in > cld_thresh) {
    if (cldn_col_in - cldn_col_in_kp1 > cld_thresh) {

      // cloud base

      // rce-comments
      // first, should probably have 1/zs(k) here rather than dz(i,k)
      // because the turbulent flux is proportional to eddy_diff(k)*zs(k),
      // while the dz(i,k) is used to get flux divergences and mixing
      // ratio tendency/change second and more importantly, using a single
      // updraft velocity here means having monodisperse turbulent
      // updraft and downdrafts. The sq2pi factor assumes a normal draft
      // spectrum. The fluxn/fluxm from activate must be consistent with
      // the fluxes calculated in explmix.
      eddy_diff = wtke_col_in / zs;
      // rce-comment - use kp1 here as old-cloud activation involves
      // aerosol from layer below

      // Real fn[ntot_amode] = {};// activation fraction for aerosol number
      // [fraction]
      // activation fraction for aerosol mass [fraction]
      Real fm[ntot_amode] = {};
      // flux of activated aerosol mass fraction into cloud [m/s]
      Real fluxm[ntot_amode] = {};
      // flux of activated aerosol number fraction into cloud [m/s]
      Real fluxn[ntot_amode] = {};
      Real flux_fullact = zero;
      // flux of activated aerosol fraction assuming 100% activation [m/s]
      get_activate_frac(state_q_col_in_kp1, air_density_kp1, air_density,
                        wtke_col_in,
                        temp_col_in, // in
                        voltonumbhi_amode, voltonumblo_amode, exp45logsig,
                        alogsig, aten, factnum_col, fm, fluxn, fluxm, // out
                        flux_fullact);

      //  store for output activation fraction of aerosol
      // factnum_col(kk,:) = fn
      // vertical change in cloud fraction [fraction]
      const Real delz_cld = cldn_col_in - cldn_col_in_kp1;

      // flux of activated aerosol number into cloud [#/m^2/s]
      Real fluxntot = zero;

      // rce-comment 1
      // flux of activated mass into layer k (in kg/m2/s)
      // = "actmassflux" = dumc*fluxm*raercol(kp1,lmass)*csbot(k)
      // source of activated mass (in kg/kg/s) = flux divergence
      // = actmassflux/(cs(i,k)*dz(i,k))
      // so need factor of csbot_cscen = csbot(k)/cs(i,k)
      // dum=1./(dz(i,k))
      // conversion factor from flux to rate [m^{-1}]
      const Real crdz = csbot_cscen / dz;

      // rce-comment 2
      // code for k=pver was changed to use the following conceptual model
      // in k=pver, there can be no cloud-base activation unless one
      // considers a scenario such as the layer being partially cloudy,
      // with clear air at bottom and cloudy air at top
      // assume this scenario, and that the clear/cloudy portions mix with
      // a timescale taumix_internal = dz(i,pver)/wtke_cen(i,pver)
      // in the absence of other sources/sinks, qact (the activated
      // particle mixratio) attains a steady state value given by
      // qact_ss = fcloud*fact*qtot where fcloud is cloud fraction, fact
      // is activation fraction, qtot=qact+qint, qint is interstitial
      // particle mixratio the activation rate (from mixing within the
      // layer) can now be written as d(qact)/dt = (qact_ss -
      // qact)/taumix_internal = qtot*(fcloud*fact*wtke/dz) - qact*(wtke/dz)
      // note that (fcloud*fact*wtke/dz) is equal to the nact/mact
      // also, d(qact)/dt can be negative.
      // in the code below it is forced to be >= 0

      // steve --
      // you will likely want to change this. i did not really understand
      // what was previously being done in k=pver
      // in the cam3_5_3 code, wtke(i,pver) appears to be equal to the
      // droplet deposition velocity which is quite small
      // in the cam3_5_37 version, wtke is done differently and is much
      // larger in k=pver, so the activation is stronger there

      for (int imode = 0; imode < ntot_amode; ++imode) {
        // local array index for MAM number, species
        // Fortran indexing to C++ indexing
        const int mm = AeroConfig::mam_idx(imode, 0);
        nact[imode] += fluxn[imode] * crdz * delz_cld;
        mact[imode] += fluxm[imode] * crdz * delz_cld;
        // note that kp1 is used here
        fluxntot +=
            fluxn[imode] * delz_cld * raercol_nsav_kp1[mm] * air_density;
      } // end imode

      nsource_col += fluxntot / (air_density * dz);
    } // end cldn_col_in(kk) - cldn_col_in(kp1) > cld_thresh
  } else {

    // if cldn < 0.01_r8 at any level except kk=pver, deplete qcld,
    // turn all raercol_cw to raercol, put appropriate tendency in nsource
    // Note that if cldn(kk) >= 0.01_r8 but cldn(kk) - cldn(kp1) <= 0.01,
    // nothing is done.

    // no cloud
    nsource_col -= qcld * dtinv;
    qcld = zero;

    // convert activated aerosol to interstitial in decaying cloud
    for (int imode = 0; imode < ntot_amode; ++imode) {
      // local array index for MAM number, species
      // Fortran indexing to C++ indexing
      int mm = AeroConfig::mam_idx(imode, 0);
      raercol_nsav[mm] += raercol_cw_nsav[mm]; // cloud-borne aerosol
      raercol_cw_nsav[mm] = zero;

      for (int lspec = 1; lspec < AeroConfig::nspec_amode(imode) + 1; ++lspec) {
        mm = AeroConfig::mam_idx(imode, lspec);
        raercol_nsav[mm] += raercol_cw_nsav[mm]; // cloud-borne aerosol
        raercol_cw_nsav[mm] = zero;
      }
    }

  } // end cldn_col_in(kk) > cld_thresh
} // end update_from_cldn_profile

template <class ViewType>
KOKKOS_INLINE_FUNCTION void update_from_newcld(
    const Real cldn_col_in, const Real cldo_col_in,
    const Real dtinv, // in
    const Real wtke_col_in, const Real temp_col_in, const Real air_density,
    const Real state_q_col_in[aero_model::pcnst], // in
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const Real exp45logsig[AeroConfig::num_modes()],
    const Real alogsig[AeroConfig::num_modes()], const Real aten, Real &qcld,
    ViewType raercol_nsav,
    ViewType raercol_cw_nsav, // inout
    Real &nsource_col_out, Real factnum_col_out[AeroConfig::num_modes()]) {

  // input arguments
  // real(r8), intent(in) :: cldn_col_in       cloud fraction [fraction]
  // real(r8), intent(in) :: cldo_col_in       cloud fraction on previous time
  // step [fraction] real(r8), intent(in) :: dtinv             inverse time step
  // for microphysics [s^{-1}] real(r8), intent(in) :: wtke_col_in       subgrid
  // vertical velocity [m/s] real(r8), intent(in) :: temp_col_in temperature [K]
  // real(r8), intent(in) :: cs_col_in         air density at actual level kk
  // [kg/m^3] real(r8), intent(in) :: state_q_col_in(:) aerosol mmrs [kg/kg]
  //
  // real(r8), intent(inout) :: qcld               cloud droplet number mixing
  // ratio [#/kg] real(r8), intent(inout) :: raercol_nsav(:)    single column of
  // saved aerosol mass, number mixing ratios [#/kg or kg/kg] real(r8),
  // intent(inout) :: raercol_cw_nsav(:) same as raercol_nsav but for
  // cloud-borne phase [#/kg or kg/kg] real(r8), intent(inout) ::
  // nsource_col_out(:) droplet number mixing ratio source tendency [#/kg/s]
  // real(r8), intent(inout) :: factnum_col_out(:) activation fraction for
  // aerosol number [fraction]

  const Real zero = 0;
  const Real one = 1;
  constexpr int ntot_amode = AeroConfig::num_modes();
  // threshold cloud fraction growth[fraction]
  // BAD CONSTANT
  const Real grow_cld_thresh = 0.01;

  // new - old cloud fraction [fraction]
  const Real delt_cld = cldn_col_in - cldo_col_in;

  // shrinking cloud ......................................................
  // treat the reduction of cloud fraction from when cldn(i,k) < cldo(i,k)
  // and also dissipate the portion of the cloud that will be regenerated

  if (cldn_col_in < cldo_col_in) {
    //  droplet loss in decaying cloud
    // FIXME: does this mean anything? (from Fortran code)
    // ++ sungsup
    nsource_col_out += qcld * (cldn_col_in - cldo_col_in) / cldo_col_in * dtinv;

    qcld *= (one + (cldn_col_in - cldo_col_in) / cldo_col_in);
    // FIXME: does this mean anything? (from Fortran code)
    // -- sungsup
    // convert activated aerosol to interstitial in decaying cloud
    // fractional change in cloud fraction [fraction]
    const Real frac_delt_cld = (cldn_col_in - cldo_col_in) / cldo_col_in;

    for (int imode = 0; imode < ntot_amode; ++imode) {
      // Fortran indexing to C++ indexing
      const int mm = AeroConfig::mam_idx(imode, 0);
      // cloud-borne aerosol tendency due to cloud frac tendency [#/kg or kg/kg]
      const Real dact = raercol_cw_nsav[mm] * frac_delt_cld;
      raercol_cw_nsav[mm] += dact; // cloud-borne aerosol
      raercol_nsav[mm] -= dact;
      for (int lspec = 1; lspec < AeroConfig::nspec_amode(imode) + 1; ++lspec) {
        // Fortran indexing to C++ indexing
        const int mm = AeroConfig::mam_idx(imode, lspec);
        const Real dact = raercol_cw_nsav[mm] * frac_delt_cld;
        raercol_cw_nsav[mm] += dact; // cloud-borne aerosol
        raercol_nsav[mm] -= dact;
      } // lspec
    }   // imode

  } // cldn_col_in < cldo_col_in

  // growing cloud ......................................................
  // treat the increase of cloud fraction from when cldn(i,k) > cldo(i,k)
  // and also regenerate part of the cloud

  if (cldn_col_in - cldo_col_in > grow_cld_thresh) {
    Real fm[ntot_amode] = {}; // activation fraction for aerosol mass [fraction]
    // flux of activated aerosol mass fraction into cloud [m/s]
    Real fluxm[ntot_amode] = {};
    // flux of activated aerosol number fraction into cloud [m/s]
    Real fluxn[ntot_amode] = {};
    // flux of activated aerosol fraction assuming 100% activation [m/s]
    Real flux_fullact = zero;

    get_activate_frac(state_q_col_in, air_density, air_density, wtke_col_in,
                      temp_col_in, // in
                      voltonumbhi_amode, voltonumblo_amode, exp45logsig,
                      alogsig, aten, factnum_col_out, fm, fluxn, fluxm, // out
                      flux_fullact);

    for (int imode = 0; imode < ntot_amode; ++imode) {
      // Fortran indexing to C++ indexing
      const int mm = AeroConfig::mam_idx(imode, 0);
      // Fortran indexing to C++ indexing
      const int num_idx = AeroConfig::numptr_amode(imode);
      const Real dact = delt_cld * factnum_col_out[imode] *
                        state_q_col_in[num_idx]; // interstitial only
      qcld += dact;
      nsource_col_out += dact * dtinv;
      raercol_cw_nsav[mm] += dact; // cloud-borne aerosol
      raercol_nsav[mm] -= dact;
      // fm change from fractional change in cloud fraction [fraction]
      const Real fm_delt_cld = delt_cld * fm[imode];

      for (int lspec = 1; lspec < AeroConfig::nspec_amode(imode) + 1; ++lspec) {
        // Fortran indexing to C++ indexing
        const int mm = AeroConfig::mam_idx(imode, lspec);
        // Fortran indexing to C++ indexing
        const int spc_idx = AeroConfig::lmassptr_amode(lspec - 1, imode);
        // interstitial only
        const Real dact = fm_delt_cld * state_q_col_in[spc_idx];
        raercol_cw_nsav[mm] += dact; //  cloud-borne aerosol
        raercol_nsav[mm] -= dact;

      } // lspec
    }   // imode
  }     // cldn_col_in - cldo_col_in  > grow_cld_thresh

} // update_from_newcld

// explmix output: updated number or mass mixing ratio [# or kg / kg] depending
// on inputs
KOKKOS_INLINE_FUNCTION
Real explmix(const Real qold_km1, // number / mass mixing ratio from previous
                                  // time step at level k-1 [# or kg / kg]
             const Real qold_k, // number / mass mixing ratio from previous time
                                // step at level k [# or kg / kg]
             const Real qold_kp1, // number / mass mixing ratio from previous
                                  // time step at level k+1 [# or kg / kg]
             const Real src, // source due to activation/nucleation at level k
                             // [# or kg / (kg-s)]
             const Real eddy_diff_kp, // zn*zs*density*diffusivity (kg/m3 m2/s)
                                      // at interface
                                      // [/s]; below layer k  (k,k+1 interface)
             const Real eddy_diff_km, // zn*zs*density*diffusivity (kg/m3 m2/s)
                                      // at interface
                                      // [/s]; above layer k  (k,k+1 interface)
             const Real overlapp,     // cloud overlap below [fraction]
             const Real overlapm,     // cloud overlap above [fraction]
             const Real dtmix         // time step [s]
) {
  Real qnew =
      qold_k + dtmix * (src + eddy_diff_kp * (overlapp * qold_kp1 - qold_k) +
                        eddy_diff_km * (overlapm * qold_km1 - qold_k));
  // force to non-negative
  qnew = mam4::max(qnew, 0);
  // output: updated number or mass mixing ratio [# or kg / kg]
  return qnew;
} // end explmix

// explmix output: updated number or mass mixing ratio [# or kg / kg] depending
// on inputs
KOKKOS_INLINE_FUNCTION
Real explmix(const Real qold_km1, // number / mass mixing ratio from previous
                                  // time step at level k-1 [# or kg / kg]
             const Real qold_k, // number / mass mixing ratio from previous time
                                // step at level k [# or kg / kg]
             const Real qold_kp1, // number / mass mixing ratio from previous
                                  // time step at level k+1 [# or kg / kg]
             const Real src, // source due to activation/nucleation at level k
                             // [# or kg / (kg-s)]
             const Real eddy_diff_kp, // zn*zs*density*diffusivity (kg/m3 m2/s)
                                      // at interface
                                      // [/s]; below layer k  (k,k+1 interface)
             const Real eddy_diff_km, // zn*zs*density*diffusivity (kg/m3 m2/s)
                                      // at interface
                                      // [/s]; above layer k  (k,k+1 interface)
             const Real overlapp,     // cloud overlap below [fraction]
             const Real overlapm,     // cloud overlap above [fraction]
             const Real dtmix,        // time step [s]
             const Real qactold_km1,
             // optional: number / mass mixing ratio of ACTIVATED species
             // from previous step at level k-1 *** this should only be present
             // if the current species is unactivated number/sfc/mass
             const Real qactold_kp1
             // optional: number / mass mixing ratio of ACTIVATED species
             // from previous step at level k+1 *** this should only be present
             // if the current species is unactivated number/sfc/mass
) {

  // the qactold*(1-overlap) terms are resuspension of activated material
  const Real one = 1.0;
  Real qnew =
      qold_k +
      dtmix *
          (-src +
           eddy_diff_kp * (qold_kp1 - qold_k + qactold_kp1 * (one - overlapp)) +
           eddy_diff_km * (qold_km1 - qold_k + qactold_km1 * (one - overlapm)));

  // force to non-negative
  qnew = mam4::max(qnew, 0);
  // output: updated number or mass mixing ratio [# or kg / kg]
  return qnew;
} // end explmix

KOKKOS_INLINE_FUNCTION
void update_from_explmix(
    const ThreadTeam &team,
    const Real dtmicro, // time step for microphysics [s]
    const ColumnView
        &csbot, // air density at bottom (interface) of layer [kg/m^3]
    const ConstColumnView &cldn, // cloud fraction [fraction]
    const ColumnView &zn,        // g/pdel for layer [m^2/kg]
    const ColumnView &zs,        // inverse of distance between levels [m^-1]
    const ColumnView &eddy_diff, // diffusivity for droplets [m^2/s]
    const View2D &nact,          // fractional aero. number activation rate [/s]
    const View2D &mact,          // fractional aero. mass activation rate [/s]
    const ColumnView &qcld,      // cloud droplet number mixing ratio [#/kg]
    // single column of saved aerosol mass, number mixing ratios [#/kg or kg/kg]
    const View3D raercol,
    // same as raercol but for cloud-borne phase [#/kg or kg/kg]
    const View3D raercol_cw,
    int &nsav, // indices for old, new time levels in substepping
    int &nnew, // indices for old, new time levels in substepping
    const int top_lev,
    // work vars
    const ColumnView &overlapp, // cloud overlap involving level kk+1 [fraction]
    const ColumnView &overlapm, // cloud overlap involving level kk-1 [fraction]
    const ColumnView &eddy_diff_kp, // zn*zs*density*diffusivity [/s]
    const ColumnView &eddy_diff_km, // zn*zs*density*diffusivity   [/s]
    const ColumnView
        &qncld // work space: stores qcld so qcld can be updated in parallal
) {

  // threshold cloud fraction to compute overlap [fraction]
  // BAD CONSTANT
  const Real overlap_cld_thresh = 1e-10;
  const Real zero = 0.0;
  const Real one = 1.0;

  constexpr int ntot_amode = AeroConfig::num_modes();
  // load new droplets in layers above, below clouds
  Real dtmin = dtmicro;
  // rce-comment -- eddy_diff(k) is eddy-diffusivity at k/k+1 interface
  // want eddy_diff_k(k) = eddy_diff(k) * (density at k/k+1 interface)
  // so use pint(i,k+1) as pint is 1:pverp
  // eddy_diff_k(k)=eddy_diff(k)*2.*pint(i,k)/(rair*(temp(i,k)+temp(i,k+1)))
  // eddy_diff_k(k)=eddy_diff(k)*2.*pint(i,k+1)/(rair*(temp(i,k)+temp(i,k+1)))

  // start k for loop here. for k = top_lev to pver
  // cldn will be columnviews of length pver,
  // overlaps also to columnview pass as parameter so it is allocated elsewhere
  static constexpr int pver_loc = pver;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, top_lev, pver_loc),
      [&](int k, Real &min_val) {
        const int kp1 = mam4::min(k + 1, pver_loc - 1);
        const int km1 = mam4::max(k - 1, top_lev);
        // maximum overlap assumption
        if (cldn(kp1) > overlap_cld_thresh) {
          overlapp(k) = mam4::min(cldn(k) / cldn(kp1), one);
        } else {
          overlapp(k) = one;
        }

        if (cldn(km1) > overlap_cld_thresh) {
          overlapm(k) = mam4::min(cldn(k) / cldn(km1), one);
        } else {
          overlapm(k) = one;
        }

        eddy_diff_kp(k) = zn(k) * eddy_diff(k) * csbot(k) * zs(k);
        // NOTE: eddy_diff_k uses k-1 while sz uses km1.
        eddy_diff_km(k) = zn(k) * eddy_diff(km1) * csbot(km1) * zs(km1);
        const Real tinv = eddy_diff_kp(k) + eddy_diff_km(k);

        // rce-comment
        // the activation source(k) = mact(k,m)*raercol(lmass, kp1)
        // should not exceed the rate of transfer of unactivated particles
        // from kp1 to k which = eddy_diff_kp(k)*raercol(lmass, kp1)
        // however it might if things are not "just right" in subr activate
        // the following is a safety measure to avoid negatives in explmix

        for (int imode = 0; imode < ntot_amode; imode++) {
          nact(k, imode) = mam4::min(nact(k, imode), eddy_diff_kp(k));
          mact(k, imode) = mam4::min(mact(k, imode), eddy_diff_kp(k));
        }

        // rce-comment -- tinv is the sum of all first-order-loss-rates
        // for the layer.  for most layers, the activation loss rate
        // (for interstitial particles) is accounted for by the loss by
        // turb-transfer to the layer above.
        // k=pver is special, and the loss rate for activation within
        // the layer must be added to tinv.  if not, the time step
        // can be too big, and explmix can produce negative values.
        // the negative values are reset to zero, resulting in an
        // artificial source.
        //  BAD CONSTANT
        if (tinv > 1e-6) {
          min_val = mam4::min(min_val, one / tinv);
        }
      },
      Kokkos::Min<Real>(dtmin));
  team.team_barrier();

  // timescale for subloop [s]
  //  BAD CONSTANT
  Real dtmix = 0.9 * dtmin;
  // number of substeps and bound
  const int nsubmix = dtmicro / dtmix + 1;

  dtmix = dtmicro / nsubmix;

  // old_cloud_nsubmix_loop
  //  Note: each pass in submix loop stores updated aerosol values at index
  //  nnew, current values at index nsav. At the start of each pass, nnew
  //  values are copied to nsav. However, this is accomplished by switching the
  //  values of nsav and nnew rather than a physical copying. At end of loop
  //  nnew stores index of most recent updated values (either 1 or 2).

  team.team_barrier();
  for (int isub = 0; isub < nsubmix; isub++) {
    if (0 < isub) {
      // after first pass, switch nsav, nnew so that nsav is the
      // recently updated aerosol
      const int ntemp = nsav;
      nsav = nnew;
      nnew = ntemp;
    }
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, top_lev, pver_loc),
                         [&](int k) { qncld(k) = qcld(k); });
    team.team_barrier();
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team, top_lev, pver_loc), [&](int k) {
          const int kp1 = mam4::min(k + 1, pver_loc - 1);
          const int km1 = mam4::max(k - 1, top_lev);
          const auto raercol_km1_nsav =
              Kokkos::subview(raercol, nsav, Kokkos::ALL(), km1);
          const auto raercol_k_nsav =
              Kokkos::subview(raercol, nsav, Kokkos::ALL(), k);
          const auto raercol_kp1_nsav =
              Kokkos::subview(raercol, nsav, Kokkos::ALL(), kp1);
          const auto raercol_k_nnew =
              Kokkos::subview(raercol, nnew, Kokkos::ALL(), k);
          const auto raercol_cw_km1_nsav =
              Kokkos::subview(raercol_cw, nsav, Kokkos::ALL(), km1);
          const auto raercol_cw_k_nsav =
              Kokkos::subview(raercol_cw, nsav, Kokkos::ALL(), k);
          const auto raercol_cw_kp1_nsav =
              Kokkos::subview(raercol_cw, nsav, Kokkos::ALL(), kp1);
          const auto raercol_cw_k_nnew =
              Kokkos::subview(raercol_cw, nnew, Kokkos::ALL(), k);
          // update droplet source
          // rce-comment- activation source in layer k involves particles from
          // k+1
          //         srcn(:)=srcn(:)+nact(:,m)*(raercol(:,mm,nsav))
          Real srcn = zero;
          for (int imode = 0; imode < ntot_amode; imode++) {
            const int mm = AeroConfig::mam_idx(imode, 0);
            srcn += nact(k, imode) * raercol_kp1_nsav(mm);
            if (k == pver_loc - 1) {
              // rce-comment- new formulation for k=pver
              // srcn(  pver  )=srcn(  pver  )+nact(  pver
              // ,m)*(raercol(pver,mm,nsav))
              const Real tmpa = raercol_k_nsav(mm) * nact(k, imode) +
                                raercol_cw_k_nsav(mm) * nact(k, imode);
              srcn += mam4::max(zero, tmpa);
            }
          } // end imode
          // update aerosol number
          // rce-comment
          //    the interstitial particle mixratio is different in clear/cloudy
          //    portions of a layer, and generally higher in the clear portion.
          //    (we have/had a method for diagnosing the the clear/cloudy
          //    mixratios.) the activation source terms involve clear air (from
          //    below) moving into cloudy air (above). in theory, the
          //    clear-portion mixratio should be used when calculating source
          //    terms
          // rce-comment: activation source in layer k involves particles from
          // k+1 source(:)= mact(:,m)*(raercol(:,mm,nsav))
          qcld(k) =
              explmix(qncld(km1), qncld(k), qncld(kp1), srcn, eddy_diff_kp(k),
                      eddy_diff_km(k), overlapp(k), overlapm(k), dtmix);
          for (int imode = 0; imode < ntot_amode; imode++) {
            for (int lspec = 0; lspec < AeroConfig::nspec_amode(imode) + 1;
                 lspec++) {
              const int mm = AeroConfig::mam_idx(imode, lspec);
              Real source = 0;
              if (k < pver_loc - 1) {
                const Real act = lspec ? mact(k, imode) : nact(k, imode);
                source = act * raercol_kp1_nsav(mm);
              } else {
                const Real tmpa = raercol_k_nsav(mm) * nact(k, imode) +
                                  raercol_cw_k_nsav(mm) * nact(k, imode);
                source = mam4::max(zero, tmpa);
              }
              // update aerosol species mass
              raercol_cw_k_nnew(mm) =
                  explmix(raercol_cw_km1_nsav(mm), raercol_cw_k_nsav(mm),
                          raercol_cw_kp1_nsav(mm), source, eddy_diff_kp(k),
                          eddy_diff_km(k), overlapp(k), overlapm(k), dtmix);
              raercol_k_nnew(mm) =
                  explmix(raercol_km1_nsav(mm), raercol_k_nsav(mm),
                          raercol_kp1_nsav(mm), source, eddy_diff_kp(k),
                          eddy_diff_km(k), overlapp(k), overlapm(k), dtmix,
                          raercol_cw_km1_nsav(mm), raercol_cw_kp1_nsav(mm));
            } // lspec loop
          }   // imode loop
        });   // k loop
    team.team_barrier();
  } // old_cloud_nsubmix_loop

  // evaporate particles again if no cloud
  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, top_lev, pver_loc), [&](int k) {
        if (cldn(k) == zero) {
          // no cloud
          qcld(k) = zero;

          // convert activated aerosol to interstitial in decaying cloud
          for (int imode = 0; imode < ntot_amode; imode++) {
            const int mm = AeroConfig::mam_idx(imode, 0);
            raercol(nnew, mm, k) += raercol_cw(nnew, mm, k);
            raercol_cw(nnew, mm, k) = zero;

            for (int lspec = 1; lspec < AeroConfig::nspec_amode(imode) + 1;
                 lspec++) {
              const int mm = AeroConfig::mam_idx(imode, lspec);
              raercol(nnew, mm, k) += raercol_cw(nnew, mm, k);
              raercol_cw(nnew, mm, k) = zero;
            } // lspec
          }   // imode
        }     // if cldn(k) == 0
      });     // kk
} // end update_from_explmix

KOKKOS_INLINE_FUNCTION
void dropmixnuc(
    const ThreadTeam &team, const Real dtmicro, const ConstColumnView &temp,
    const ConstColumnView &pmid, const ConstColumnView &pint,
    const ConstColumnView &pdel, const ConstColumnView &rpdel,
    const ConstColumnView &zm, const ConstView2D &state_q,
    const ConstColumnView &ncldwtr, const ConstColumnView &v_diffusivity,
    const ConstColumnView &cldn,
    const Real voltonumbhi_amode[AeroConfig::num_modes()],
    const Real voltonumblo_amode[AeroConfig::num_modes()],
    const Real exp45logsig[AeroConfig::num_modes()],
    const Real alogsig[AeroConfig::num_modes()], const Real aten, const bool &,
    const ColumnView &qcld, const ColumnView &wsub,
    const ColumnView &cldo, // in
    const View2D qqcw_fld,  // inout
    const View2D ptend_q, const ColumnView &tendnd, const View2D &factnum,
    const ColumnView &ndropcol, const ColumnView &ndropmix,
    const ColumnView &nsource, const ColumnView &wtke, const View2D &ccn,
    const View2D coltend, const View2D coltend_cw, const int top_lev,
    // work arrays
    const View3D raercol_cw, const View3D raercol, const View2D &nact,
    const View2D &mact, const ColumnView &eddy_diff, const ColumnView &zn,
    const ColumnView &csbot, const ColumnView &zs, const ColumnView &overlapp,
    const ColumnView &overlapm, const ColumnView &eddy_diff_kp,
    const ColumnView &eddy_diff_km, const ColumnView &qncld,
    const ColumnView &srcn, const ColumnView &source, const ColumnView &dz,
    const ColumnView &csbot_cscen, const ColumnView &raertend,
    const ColumnView &qqcwtend) {
  // vertical diffusion and nucleation of cloud droplets
  // assume cloud presence controlled by cloud fraction
  // doesn't distinguish between warm, cold clouds

  // clang-format off
  // input arguments
  // lchnk               chunk identifier
  // ncol                number of columns
  // psetcols            maximum number of columns
  // dtmicro     time step for microphysics [s]
  // temp(:,:)    temperature [K]
  // pmid(:,:)    mid-level pressure [Pa]
  // pint(:,:)    pressure at layer interfaces [Pa]
  // pdel(:,:)    pressure thickness of layer [Pa]
  // rpdel(:,:)   inverse of pressure thickness of layer [/Pa]
  // zm(:,:)      geopotential height of level [m]
  // state_q(:,:,:) aerosol mmrs [kg/kg]
  // ncldwtr(:,:) initial droplet number mixing ratio [#/kg]
  // v_diffusivity(:,:)     vertical diffusivity [m^2/s]
  // wsub(pcols,pver)    subgrid vertical velocity [m/s]
  // cldn(pcols,pver)    cloud fraction [fraction]
  // cldo(pcols,pver)    cloud fraction on previous time step [fraction]

  // inout arguments
  //  qqcw(:)     cloud-borne aerosol mass, number mixing ratios [#/kg or kg/kg]

  // output arguments
  // ptend_q
  // tendnd(pcols,pver) tendency in droplet number mixing ratio [#/kg/s]
  // factnum(:,:,:)     activation fraction for aerosol number [fraction]
  // nsource            droplet number mixing ratio source tendency [#/kg/s]
  // ndropmix           droplet number mixing ratio tendency due to mixing [#/kg/s]
  // ccn                number conc of aerosols activated at supersat [#/m^3]
  //      note:  activation fraction fluxes are defined as
  //     fluxn = [flux of activated aero. number into cloud [#/m^2/s]]
  //           / [aero. number conc. in updraft, just below cloudbase [#/m^3]]
  // coltend(:,:)       column tendency for diagnostic output
  // coltend_cw(:,:)    column tendency
  // clang-format on

  // BAD CONSTANT
  const Real zkmin = 0.01;
  const Real zkmax = 100;   // min, max vertical diffusivity [m^2/s]
  const Real wmixmin = 0.1; // minimum turbulence vertical velocity [m/s]

  const Real zero = 0;
  const Real one = 1;
  const Real two = 2;

  /// inverse time step for microphysics [s^-1]
  const Real dtinv = one / dtmicro;
  constexpr int ntot_amode = AeroConfig::num_modes();

  // NOTE FOR C++ PORT: Get the cloud borne MMRs from AD in variable qcldbrn,
  // do not port the code before END NOTE
  const Real gravity = Constants::gravity;
  const Real rair = Constants::r_gas_dry_air;

  // air_density [kg/m3]
  // geometric thickness of layers [m]
  // dz
  // fractional aero. number  activation rate [/s]
  // raercol[2][ncnst_tot][nlevels]
  // same as raercol but for cloud-borne phase [#/kg or kg/kg]
  // raercol_cw[2][ncnst_tot][nlevels]

  static constexpr int pver_loc = pver;
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 1, pver_loc), [&](int k) {
    EKAT_KERNEL_ASSERT_MSG(0 < zm(k - 1) - zm(k),
                           "Error: Geopotential height at level should be "
                           "monotonically decreasing.\n");
  });
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, pver_loc), [&](int k) {
    for (int imode = 0; imode < ntot_amode; ++imode) {
      nact(k, imode) = zero;
      mact(k, imode) = zero;
    }

    const Real cs = conversions::density_of_ideal_gas(temp(k), pmid(k));
    dz(k) = one / (cs * gravity * rpdel(k)); // layer thickness [m]
    zn(k) = gravity * rpdel(k);
    wtke(k) = mam4::max(wsub(k), wmixmin);
    // cloud droplet number mixing ratio [#/kg]
    // load number nucleated into qcld on cloud boundaries
    qcld(k) = ncldwtr(k);
    EKAT_KERNEL_ASSERT_MSG(dz(k) != 0.0, "Error: dz is equal to zero.\n");
  });

  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, top_lev, pver_loc - 1), [&](int k) {
        const Real cs = conversions::density_of_ideal_gas(temp(k), pmid(k));
        zs(k) = one / (zm(k) - zm(k + 1));
        eddy_diff(k) = utils::min_max_bound(zkmin, zkmax, v_diffusivity(k + 1));
        csbot(k) = two * pint(k + 1) / (rair * (temp(k) + temp(k + 1)));
        csbot_cscen(k) = csbot(k) / cs;
      });

  constexpr int surface_cell = pver_loc - 1;
  EKAT_KERNEL_ASSERT_MSG(
      zm(surface_cell - 1) - zm(surface_cell) != 0.0,
      "Error: zm(surface_cell - 1) - zm(surface_cell) is equal to zero.\n");
  zs(surface_cell) = one / (zm(surface_cell - 1) - zm(surface_cell));
  eddy_diff(surface_cell) = zero;
  csbot(surface_cell) =
      conversions::density_of_ideal_gas(temp(surface_cell), pmid(surface_cell));
  csbot_cscen(surface_cell) = one;
  // Initialize 1D (in space) versions of interstitial and cloud borne aerosol
  team.team_barrier(); // for wtke
  int nsav = 0;
  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, top_lev, pver_loc), [&](int k) {
        for (int imode = 0; imode < ntot_amode; ++imode) {
          const int mm = AeroConfig::mam_idx(imode, 0);
          raercol_cw(nsav, mm, k) = qqcw_fld(mm, k);
          const int num_idx = AeroConfig::numptr_amode(imode);
          raercol(nsav, mm, k) = state_q(k, num_idx);
          for (int lspec = 1; lspec < AeroConfig::nspec_amode(imode) + 1;
               ++lspec) {
            const int mm = AeroConfig::mam_idx(imode, lspec);

            raercol_cw(nsav, mm, k) = qqcw_fld(mm, k);
            const int spc_idx = AeroConfig::lmassptr_amode(lspec - 1, imode);
            raercol(nsav, mm, k) = state_q(k, spc_idx);
          } // lspec
        }   // imode

        // PART I:  changes of aerosol and cloud water from temporal changes in
        // cloud fraction droplet nucleation/aerosol activation
        nsource(k) = zero;
        const auto state_q_k = ekat::subview(state_q, k);

        Real factnum_k[ntot_amode];
        for (int imode = 0; imode < ntot_amode; ++imode)
          factnum_k[imode] = factnum(imode, k);

        // FIXME: It is dangerous to call data() on a view and expect the
        // resulting vector to be continuous in memory. Depending on the
        // 2D layout, the memory could be strided.
        auto raercol_k = Kokkos::subview(raercol, nsav, Kokkos::ALL(), k);
        auto raercol_cw_k = Kokkos::subview(raercol_cw, nsav, Kokkos::ALL(), k);
        update_from_newcld(cldn(k), cldo(k), dtinv, // in
                           wtke(k), temp(k),
                           conversions::density_of_ideal_gas(temp(k), pmid(k)),
                           state_q_k.data(), // in
                           voltonumbhi_amode, voltonumblo_amode, exp45logsig,
                           alogsig, aten, qcld(k),
                           raercol_k,              // inout
                           raercol_cw_k,           // inout
                           nsource(k), factnum_k); // inout

        for (int imode = 0; imode < ntot_amode; ++imode)
          factnum(imode, k) = factnum_k[imode];
      }); // end k
          // end k

  team.team_barrier();
  // NOTE: update_from_cldn_profile loops from 7 to 71 in fortran code.
  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, top_lev, pver_loc - 1), [&](int k) {
        const int kp1 = mam4::min(k + 1, pver - 1);

        // PART II: changes in aerosol and cloud water from vertical profile of
        // new cloud fraction
        const auto state_q_kp1 = ekat::subview(state_q, kp1);
        Real factnum_k[ntot_amode];
        for (int imode = 0; imode < ntot_amode; ++imode)
          factnum_k[imode] = factnum(imode, k);
        const auto nact_k = ekat::subview(nact, k);
        const auto mact_k = ekat::subview(mact, k);
        const auto raercol_k = Kokkos::subview(raercol, nsav, Kokkos::ALL(), k);
        const auto raercol_kp1 =
            Kokkos::subview(raercol, nsav, Kokkos::ALL(), kp1);
        const auto raercol_cw_k =
            Kokkos::subview(raercol_cw, nsav, Kokkos::ALL(), k);
        update_from_cldn_profile(
            cldn(k), cldn(kp1), dtinv, wtke(k), zs(k), dz(k), // in
            temp(k), conversions::density_of_ideal_gas(temp(k), pmid(k)),
            conversions::density_of_ideal_gas(temp(kp1), pmid(kp1)),
            csbot_cscen(k),
            state_q_kp1.data(), // in
            voltonumbhi_amode, voltonumblo_amode, exp45logsig, alogsig, aten,
            raercol_k, raercol_kp1, raercol_cw_k,
            nsource(k), // inout
            qcld(k), factnum_k,
            eddy_diff(k), // out
            nact_k.data(), mact_k.data());
        for (int imode = 0; imode < ntot_amode; ++imode)
          factnum(imode, k) = factnum_k[imode];
      });
  team.team_barrier();

  // PART III:  perform explicit integration of droplet/aerosol mixing using
  // substepping

  int nnew = 1;
  update_from_explmix(team, dtmicro, csbot, cldn, zn, zs, eddy_diff, nact, mact,
                      qcld, raercol, raercol_cw, nsav, nnew, top_lev,
                      // work vars
                      overlapp, overlapm, eddy_diff_kp, eddy_diff_km, qncld);

  team.team_barrier();

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, top_lev), [&](int kk) {
    for (int i = 0; i < ncnst_tot; ++i) {
      qqcw_fld(i, kk) = zero;
    }
  });
  team.team_barrier();

  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, top_lev, pver_loc), [&](int k) {
        // droplet number mixing ratio tendency due to mixing [#/kg/s]
        ndropmix(k) = (qcld(k) - ncldwtr(k)) * dtinv - nsource(k);
        // BAD CONSTANT
        tendnd(k) = (max(qcld(k), 1.e-6) - ncldwtr(k)) * dtinv;
        // ndropcol(icol) = ndropcol(icol)/gravity
        // sum up ndropcol_kk outside of kk loop
        // column-integrated droplet number [#/m2]
        ndropcol(k) = ncldwtr(k) * pdel(k) / gravity;
        // tendency of interstitial aerosol mass, number mixing ratios
        // [#/kg/s] or [kg/kg/s]
        raertend(k) = zero;
        // tendency of cloudborne aerosol mass, number mixing ratios
        // [#/kg/s]or [kg/kg/s]
        qqcwtend(k) = zero;
        // cloud-borne aerosol mass mixing ratios [kg/kg]
        Real qcldbrn[maxd_aspectype][ntot_amode] = {{zero}};
        Real qcldbrn_num[ntot_amode] = {zero};

        for (int imode = 0; imode < ntot_amode; ++imode) {
          // species index for given mode
          for (int lspec = 0; lspec < AeroConfig::nspec_amode(imode) + 1;
               ++lspec) {
            // local array index for MAM number, species
            // Fortran indexing to C++ indexing
            const int mm = AeroConfig::mam_idx(imode, lspec);
            // Fortran indexing to C++ indexing
            const int lptr = AeroConfig::mam_cnst_idx(imode, lspec);
            qqcwtend(k) = (raercol_cw(nnew, mm, k) - qqcw_fld(mm, k)) * dtinv;
            qqcw_fld(mm, k) = mam4::max(raercol_cw(nnew, mm, k),
                                        zero); // update cloud-borne aerosol

            if (lspec == 0) {
              // Fortran indexing to C++ indexing
              const int num_idx = AeroConfig::numptr_amode(imode);
              raertend(k) =
                  (raercol(nnew, mm, k) - state_q(k, num_idx)) * dtinv;
              qcldbrn_num[imode] = qqcw_fld(mm, k);
            } else {
              // Fortran indexing to C++ indexing
              const int spc_idx = AeroConfig::lmassptr_amode(lspec - 1, imode);
              raertend(k) =
                  (raercol(nnew, mm, k) - state_q(k, spc_idx)) * dtinv;
              // Extract cloud borne MMRs from qqcw pointer
              qcldbrn[lspec][imode] = qqcw_fld(mm, k);
            } // end if
            // NOTE: perform sum after loop. Thus, we need to store coltend_kk
            // and coltend_cw_kk Port this code outside of this function
            coltend(mm, k) = pdel(k) * raertend(k) / gravity;
            coltend_cw(mm, k) = pdel(k) * qqcwtend(k) / gravity;
            // set tendencies for interstitial aerosol
            ptend_q(lptr, k) = raertend(k);

          } // lspec
        }   // imode

        const auto state_q_k = ekat::subview(state_q, k);
        const auto ccn_k = ekat::subview(ccn, k);

        //  Use interstitial and cloud-borne aerosol to compute output
        // ccn fields.
        ccncalc(state_q_k.data(), temp(k), qcldbrn, qcldbrn_num,
                conversions::density_of_ideal_gas(temp(k), pmid(k)),
                voltonumbhi_amode, voltonumblo_amode, exp45logsig, alogsig,
                ccn_k.data());
      }); // end parfor(k)
  team.team_barrier();
} // dropmixnuc
} // namespace ndrop
} // end namespace mam4

#endif
