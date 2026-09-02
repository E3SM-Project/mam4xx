// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_WET_DEPOSITION_HPP
#define MAM4XX_WET_DEPOSITION_HPP

#include "aero_config.hpp"
#include "aero_model.hpp"
#include "atmosphere.hpp"
#include "mam4_constants.hpp"
#include "mam4_math.hpp"
#include "modal_aero_calcsize.hpp"
#include "utils.hpp"
#include "water_uptake.hpp"

#include <ekat_subview_utils.hpp>

// Based on e3sm_mam4_refactor/components/eam/src/chemistry/aerosol/wetdep.F90
namespace mam4 {

namespace wetdep {

/**
 * @brief Calculate local precipitation generation rate (kg/m2/s)
 *        from source (condensation) and sink (evaporation) terms.
 *
 * @param[in] pdel Pressure difference across layers [Pa].
 * @param[in] source_term Precipitation source term rate (condensation)
 * [kg/kg/s].
 * @param[in] sink_term Precipitation sink term rate (evaporation) [kg/kg/s].
 * @param[in] atm Atmosphere object (used for number of levels).
 *
 * @param[out] lprec Local production rate of precipitation [kg/m2/s].
 *
 * @pre pdel, source_term, sink_term and lprec are all an array
 *      of size pver == atm.num_levels().
 *
 * @pre In F90, ncol == 1 as we only operate over one column at a time
 *      as outer loops will iterate over columns, so we drop ncol as input.
 * @pre In F90 code, pcols is the number of columns in the mesh.
 *      Since we are only operating over one column, pcols == 1.
 * @pre In F90 code, ncol == pcols. Since ncol == 1, pcols == 1.
 *
 * @pre atm is initialized correctly and has the correct number of levels.
 */
using CalcsizeData = modal_aero_calcsize::CalcsizeData;
using ConstView1D = DeviceType::view_1d<const Real>;
using View1D = DeviceType::view_1d<Real>;
using Int1D = DeviceType::view_1d<int>;
using View2D = DeviceType::view_2d<Real>;
using View2DHost = typename HostType::view_2d<Real>;
KOKKOS_INLINE_FUNCTION
Real local_precip_production(Real pdel, Real source_term, Real sink_term,
                             Real gravity) {
  return (pdel / gravity) * (source_term - sink_term);
}

// Function to call to initialize the arrays passed to the aero_model_wetdep
// function. This is a host function, so scavimptblvol and scavimptblnum need
// to be synced to device.
inline void init_scavimptbl(const AeroConfig &aero_config,
                            View2DHost scavimptblvol,
                            View2DHost scavimptblnum) {
  constexpr int num_modes = AeroConfig::num_modes();
  Real dgnum_amode[num_modes];
  Real sigmag_amode[num_modes];
  Real aerosol_dry_density[num_modes];
  for (int i = 0; i < num_modes; ++i) {
    dgnum_amode[i] = modes(i).nom_diameter;
    sigmag_amode[i] = modes(i).mean_std_dev;
  }

  auto aero_species = Kokkos::create_mirror(aero_config.aero_species);
  Kokkos::deep_copy(aero_species, aero_config.aero_species);

  // Note: Original code uses the following aerosol densities.
  // sulfate, sulfate, dust, p-organic
  aerosol_dry_density[0] = aero_species(int(AeroId::SO4)).density;
  aerosol_dry_density[1] = aero_species(int(AeroId::SO4)).density;
  aerosol_dry_density[2] = aero_species(int(AeroId::DST)).density;
  aerosol_dry_density[3] = aero_species(int(AeroId::POM)).density;
  aero_model::modal_aero_bcscavcoef_init(dgnum_amode, sigmag_amode,
                                         aerosol_dry_density, scavimptblnum,
                                         scavimptblvol);
}

// clang-format off
/**
 * @brief Calculate cloudy volume which is occupied by rain or cloud water as the
 *        max between the local cloud amount or the:
 *          sum above of (cloud * positive precip production)      sum total precip from above
 *            ---------------------------------------------    X    -------------------------
 *                   sum above of ( positive precip )             sum positive precip from above
 *
 * @param[in] cld Cloud faction [fraction, unitless].
 * @param[in] lprec Local production rate of precipitation [kg/m2/s].
 * @param[in] is_tot_cld When is_tot_cld is true, this function computes the
 *                       total cloud volume. Otherwise, it computes the cloud
 *                       volume of either the stratiform clouds or the
 *                       convective clouds (depending upon the provided cloud fraction).
 * @param[in] atm Atmosphere object (used for number of levels).
 *
 * @param[out] cldv Fraction occupied by rain or cloud water [fraction, unitless].
 *
 * @pre cld, lprec, cldv are all an array
 *      of size nlev == atm.num_levels().
 *
 * @pre In F90, ncol == 1 as we only operate over one column at a time
 *      as outer loops will iterate over columns, so we drop ncol as input.
 * @pre In F90 code, pcols is the number of columns in the mesh.
 *      Since we are only operating over one column, pcols == 1.
 * @pre In F90 code, ncol == pcols. Since ncol == 1, pcols == 1.
 *
 * @pre atm is initialized correctly and has the correct number of levels.
 *
 */
// clang-format on
template <typename FUNC>
KOKKOS_INLINE_FUNCTION void
calculate_cloudy_volume(const int nlev, const Real cld[/*nlev*/], FUNC lprec,
                        const bool is_tot_cld, Real cldv[/*nlev*/]) {
  // BAD CONSTANT
  constexpr Real small_value_30 = 1.e-30;
  constexpr Real small_value_36 = 1.e-36;
  Real sumppr = 0.0; // Precipitation rate [kg/m2/s]
  Real cldv1 = 0.0;  // Precip weighted cloud fraction from above [kg/m2/s]
  Real sumpppr = small_value_36; // Sum of positive precips from above

  for (int i = 0; i < nlev; i++) {
    const Real clouds = mam4::min(1.0, cldv1 / sumpppr) * (sumppr / sumpppr);
    if (is_tot_cld) {
      cldv[i] = mam4::max(clouds, cld[i]);
    } else {
      // For convective and stratiform precipitation volume at the top
      // interface of each layer. Neglect the current layer.
      cldv[i] = mam4::max(clouds, 0.0);
    }
    // Local production rate of precip [kg/m2/s] if positive
    const Real prec = lprec(i);
    const Real lprecp = mam4::max(prec, small_value_30);
    cldv1 += cld[i] * lprecp;
    sumppr += prec;
    sumpppr += lprecp;
  }
}

// ==============================================================================
KOKKOS_INLINE_FUNCTION
Real flux_precnum_vs_flux_prec_mpln(const Real flux_prec, const int jstrcnv) {
  // clang-format off
  // --------------------------------------------------------------------------------
  //  flux_precnum_vs_flux_prec_mp = precipitation number flux at the cloud base [drops/m^2/s]
  //  Options of assuming log-normal or marshall-palmer raindrop size distribution
  // --------------------------------------------------------------------------------
  /*
  in :: flux_prec     ! [drops/m^2/s]
  in :: jstrcnv
  out :: flux_precnum_vs_flux_prec_mpln  ! [drops/m^2/s]
  */
  // clang-format on

  // BAD CONSTANT
  constexpr Real small_value_36 = 1.e-36;

  // current only two options: 1 for marshall-palmer distribution, 2 for
  // log-normal distribution
  Real a0, a1;
  if (jstrcnv <= 1) {
    // marshall-palmer distribution
    a0 = 1.0885896550304022e+01;
    a1 = 4.3660645528167907e-01;
  } else {
    // log-normal distribution
    a0 = 9.9067806476181524e+00;
    a1 = 4.2690709912134056e-01;
  }
  Real y_var;
  if (flux_prec >= small_value_36) {
    const Real x_var = mam4::log(flux_prec);
    y_var = mam4::exp(a0 + a1 * x_var);
  } else {
    y_var = 0.0;
  }
  return y_var;
}

// ==============================================================================
KOKKOS_INLINE_FUNCTION
Real faer_resusp_vs_fprec_evap_mpln(const Real fprec_evap, const int jstrcnv) {
  // clang-format off
  //  --------------------------------------------------------------------------------
  //  corresponding fraction of precipitation-borne aerosol flux that is resuspended
  //  Options of assuming log-normal or marshall-palmer raindrop size distribution
  //  note that these fractions are relative to the cloud-base fluxes,
  //  and not to the layer immediately above fluxes
  //  --------------------------------------------------------------------------------
  /*
  in :: fprec_evap ! [fraction]
  in :: jstrcnv !  current only two options: 1 for marshall-palmer distribution, 2 for log-normal distribution
  out : faer_resusp_vs_fprec_evap_mpln ! [fraction]
  */
  // clang-format on

  // current only two options: 1 for marshall-palmer distribution, 2 for
  // log-normal distribution
  Real a01, a02, a03, a04, a05, a06, a07, a08, a09, x_lox_lin, y_lox_lin;
  if (jstrcnv <= 1) {
    // marshall-palmer distribution
    a01 = 8.6591133737322856e-02;
    a02 = -1.7389168499601941e+00;
    a03 = 2.7401882373663732e+01;
    a04 = -1.5861714653209464e+02;
    a05 = 5.1338179363011193e+02;
    a06 = -9.6835933124501412e+02;
    a07 = 1.0588489932213311e+03;
    a08 = -6.2184513459217271e+02;
    a09 = 1.5184126886039758e+02;
    x_lox_lin = 5.0000000000000003e-02;
    y_lox_lin = 2.5622471203221014e-03;
  } else {
    // log-normal distribution
    a01 = 6.1944215103685640e-02;
    a02 = -2.0095166685965378e+00;
    a03 = 2.3882460251821236e+01;
    a04 = -1.2695611774753374e+02;
    a05 = 4.0086943562320101e+02;
    a06 = -7.4954272875943707e+02;
    a07 = 8.1701055892023624e+02;
    a08 = -4.7941894659538502e+02;
    a09 = 1.1710291076059025e+02;
    x_lox_lin = 1.0000000000000001e-01;
    y_lox_lin = 6.2227889828044350e-04;
  }

  Real y_var;
  const Real x_var = utils::min_max_bound(0.0, 1.0, fprec_evap);
  if (x_var < x_lox_lin)
    y_var = y_lox_lin * (x_var / x_lox_lin);
  else
    y_var =
        x_var *
        (a01 +
         x_var *
             (a02 +
              x_var *
                  (a03 +
                   x_var *
                       (a04 +
                        x_var * (a05 +
                                 x_var * (a06 +
                                          x_var * (a07 +
                                                   x_var * (a08 +
                                                            x_var * a09))))))));

  return y_var;
}

//==============================================================================
KOKKOS_INLINE_FUNCTION
Real fprecn_resusp_vs_fprec_evap_mpln(const Real fprec_evap,
                                      const int jstrcnv) {
  // clang-format off
  // --------------------------------------------------------------------------------
  // Rain number evaporation fraction
  // Options of assuming log-normal or marshall-palmer raindrop size distribution
  // note that these fractions are relative to the cloud-base fluxes,
  // and not to the layer immediately above fluxes
  // --------------------------------------------------------------------------------
  /*
  in :: fprec_evap     ! [fraction]
  in :: jstrcnv  ! current only two options: 1 for marshall-palmer distribution, 2 for log-normal distribution
  out :: fprecn_resusp_vs_fprec_evap_mpln  ! [fraction]
  */
  // clang-format on

  // current only two options: 1 for marshall-palmer distribution, 2 for
  // log-normal distribution
  Real a01, a02, a03, a04, a05, a06, a07, a08, a09, x_lox_lin, y_lox_lin;
  if (jstrcnv <= 1) {
    // marshall-palmer distribution
    a01 = 4.5461070198414655e+00;
    a02 = -3.0381753620077529e+01;
    a03 = 1.7959619926085665e+02;
    a04 = -6.7152282193785618e+02;
    a05 = 1.5651931323557126e+03;
    a06 = -2.2743927701175126e+03;
    a07 = 2.0004645897056735e+03;
    a08 = -9.7351466279626209e+02;
    a09 = 2.0101198012962413e+02;
    x_lox_lin = 5.0000000000000003e-02;
    y_lox_lin = 1.7005858490684875e-01;
  } else {
    // log-normal distribution
    a01 = -5.2335291116884175e-02;
    a02 = 2.7203158069178226e+00;
    a03 = 9.4730878152409375e+00;
    a04 = -5.0573187592544798e+01;
    a05 = 9.4732631441282862e+01;
    a06 = -8.8265926556465814e+01;
    a07 = 3.5247835268269142e+01;
    a08 = 1.5404586576716444e+00;
    a09 = -3.8228795492549068e+00;
    x_lox_lin = 1.0000000000000001e-01;
    y_lox_lin = 2.7247994766566485e-02;
  }

  Real y_var;
  const Real x_var = utils::min_max_bound(0.0, 1.0, fprec_evap);
  if (x_var < x_lox_lin)
    y_var = y_lox_lin * (x_var / x_lox_lin);
  else
    y_var =
        x_var *
        (a01 +
         x_var *
             (a02 +
              x_var *
                  (a03 +
                   x_var *
                       (a04 +
                        x_var * (a05 +
                                 x_var * (a06 +
                                          x_var * (a07 +
                                                   x_var * (a08 +
                                                            x_var * a09))))))));

  return y_var;
}
// ==============================================================================
KOKKOS_INLINE_FUNCTION
Real wetdep_resusp_nonlinear(
    const int is_st_cu, const int mam_prevap_resusp_optcc,
    const Real precabx_old, const Real precabx_base_old, const Real scavabx_old,
    const Real precnumx_base_old, const Real precabx_new) {

  // clang-format off
  //  ------------------------------------------------------------------------------
  //  do nonlinear resuspension of aerosol mass or number
  //  ------------------------------------------------------------------------------
  /*
   in :: is_st_cu      ! options for stratiform (1) or convective (2) clouds
                       ! raindrop size distribution is
                       ! different for different cloud:
                       ! 1: assume marshall-palmer distribution
                       ! 2: assume log-normal distribution
   in :: mam_prevap_resusp_optcc       ! suspension options
   in :: precabx_base_old ! input of precipitation at cloud base [kg/m2/s]
   in :: precabx_old  ! input of precipitation above this layer [kg/m2/s]
   in :: scavabx_old  ! input scavenged tracer flux from above [kg/m2/s]
   in :: precnumx_base_old ! precipitation number at cloud base [#/m2/s]
   in :: precabx_new  ! output of precipitation above this layer [kg/m2/s]
   out :: resusp_x    ! aerosol mass re-suspension in a particular layer [kg/m2/s]
  */
  // clang-format on

  // BAD CONSTANT
  constexpr Real small_value_30 = 1.e-30;
  Real resusp_x = 0;
  // fraction of precabx and precabx_base
  const Real u_old =
      utils::min_max_bound(0.0, 1.0, precabx_old / precabx_base_old);
  // fraction after calling function *_resusp_vs_fprec_evap_mpln
  Real x_old, x_new;
  if (mam_prevap_resusp_optcc <= 130) {
    // non-linear resuspension of aerosol mass
    x_old = 1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
  } else {
    // non-linear resuspension of aerosol number based on raindrop number
    x_old = 1.0 - fprecn_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
  }
  x_old = utils::min_max_bound(0.0, 1.0, x_old);

  Real x_ratio; // fraction of x_tmp/x_old
  if (x_old < small_value_30) {
    x_new = 0.0;
    x_ratio = 0.0;
  } else {
    // fraction of precabx and precabx_base
    Real u_new = utils::min_max_bound(0.0, 1.0, precabx_new / precabx_base_old);
    u_new = mam4::min(u_new, u_old);
    if (mam_prevap_resusp_optcc <= 130) {
      // non-linear resuspension of aerosol mass
      x_new = 1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_new, is_st_cu);
    } else {
      // non-linear resuspension of aerosol number based on raindrop number
      x_new = 1.0 - fprecn_resusp_vs_fprec_evap_mpln(1.0 - u_new, is_st_cu);
    }
    x_new = utils::min_max_bound(0.0, 1.0, x_new);
    x_new = mam4::min(x_new, x_old);
    x_ratio = utils::min_max_bound(0.0, 1.0, x_new / x_old);
  }

  // update aerosol resuspension
  if (mam_prevap_resusp_optcc <= 130) {
    // aerosol mass resuspension
    const Real scavabx_new = mam4::max(0.0, scavabx_old * x_ratio);
    resusp_x = mam4::max(0.0, scavabx_old - scavabx_new);
  } else {
    // number resuspension
    resusp_x = mam4::max(0.0, precnumx_base_old * (x_old - x_new));
  }
  return resusp_x;
}
// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
Real wetdep_resusp_noprecip(const int is_st_cu,
                            const int mam_prevap_resusp_optcc,
                            const Real precabx_old, const Real precabx_base_old,
                            const Real scavabx_old,
                            const Real precnumx_base_old) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // do complete resuspension when precipitation rate is zero
  // ------------------------------------------------------------------------------
  /*
  in :: is_st_cu      ! options for stratiform (1) or convective (2) clouds
                      ! raindrop size distribution is
                      ! different for different cloud:
                      ! 1: assume marshall-palmer distribution
                      ! 2: assume log-normal distribution
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: precabx_base_old ! input of precipitation at cloud base [kg/m2/s]
  in :: precabx_old ! input of precipitation above this layer [kg/m2/s]
  in :: scavabx_old ! input of scavenged tracer flux from above [kg/m2/s]
  in :: precnumx_base_old ! precipitation number at cloud base [#/m2/s]
  out :: resusp_x    ! aerosol mass re-suspension in a particular layer [kg/m2/s]
  */
  // clang-format on

  // BAD CONSTANT
  constexpr Real small_value_30 = 1.e-30;
  Real resusp_x = 0;
  if (mam_prevap_resusp_optcc <= 130) {
    // linear resuspension based on scavenged aerosol mass or number
    resusp_x = scavabx_old;
  } else {
    if (precabx_base_old < small_value_30) {
      resusp_x = 0.0;
    } else {
      // non-linear resuspension of aerosol number based on raindrop number
      const Real u_old =
          utils::min_max_bound(0.0, 1.0, precabx_old / precabx_base_old);
      Real x_old =
          1.0 - fprecn_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
      x_old = utils::min_max_bound(0.0, 1.0, x_old);
      const Real x_new = 0.0;
      resusp_x = mam4::max(0.0, precnumx_base_old * (x_old - x_new));
    }
  }
  return resusp_x;
}
// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdep_scavenging(const int is_st_cu, const bool is_strat_cloudborne,
                       const Real deltat, const Real fracp, const Real precabx,
                       const Real cldv_ik, const Real scavcoef_ik,
                       const Real sol_factb, const Real sol_facti,
                       const Real tracer_1, const Real tracer_2, Real &src,
                       Real &fin) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // do scavenging for both convective and stratiform
  //
  // assuming liquid clouds (no ice)
  //
  // set odds proportional to fraction of the grid box that is swept by the
  // precipitation =precabc/rhoh20*(area of sphere projected on plane
  //                                /volume of sphere)*deltat
  // assume the radius of a raindrop is 1 e-3 m from Rogers and Yau,
  // unless the fraction of the area that is cloud is less than odds, in which
  // case use the cloud fraction (assumes precabs is in kg/m2/s)
  // is really: precabs*3/4/1000./1e-3*deltat
  // here I use .1 from Balkanski
  // ------------------------------------------------------------------------------
  /*
  in :: is_strat_cloudborne   ! if tracer is stratiform-cloudborne aerosol or not
  in :: is_st_cu ! options for stratiform (1) or convective (2) clouds

  in :: deltat       ! timestep [s]
  in :: fracp        ! fraction of cloud water converted to precip [fraction]
  in :: precabx      ! precip from above of the layer [kg/m2/s]
  in :: cldv_ik      ! precipitation area at the top interface [fraction]
  in :: scavcoef_ik  ! Dana and Hales coefficient [1/mm]
  in :: sol_factb    ! solubility factor (frac of aerosol scavenged below cloud) [fraction]
  in :: sol_facti    ! solubility factor (frac of aerosol scavenged in cloud) [fraction]
  in :: tracer_1     ! tracer input for calculate src1 [kg/kg]
  in :: tracer_2     ! tracer input for calculate src2 [kg/kg]
  out :: src         ! total scavenging (incloud + belowcloud) [kg/kg/s]
  out :: fin         ! fraction of incloud scavenging [fraction]

  */
  // clang-format on
  // BAD CONSTANT
  constexpr Real small_value_36 = 1.e-36;
  constexpr Real small_value_5 = 1.e-5; // for cloud fraction

  // calculate limitation of removal rate using Dana and Hales coefficient
  // odds  : limit on removal rate (proportional to prec) [fraction]
  Real odds =
      precabx / mam4::max(cldv_ik, small_value_5) * scavcoef_ik * deltat;
  odds = utils::min_max_bound(0.0, 1.0, odds);

  Real src1; // incloud scavenging tendency [kg/kg/s]
  Real src2; // below-cloud scavenging tendency [kg/kg/s]
  if (is_strat_cloudborne) {
    if (is_st_cu == 2) {
      // convective cloud does not affect strat-cloudborne aerosol
      src1 = 0;
    } else {
      // strat in-cloud removal only affects strat-cloudborne aerosol
      // in-cloud scavenging:
      src1 = (sol_facti * fracp * tracer_1) / deltat;
    }
    // no below-cloud scavenging for strat-cloudborne aerosol
    src2 = 0;
  } else {
    if (is_st_cu == 2) { // convective
      src1 = (sol_facti * fracp * tracer_1) / deltat;
    } else { // stratiform
      // strat in-cloud removal only affects strat-cloudborne aerosol
      src1 = 0;
    }
    src2 = (sol_factb * cldv_ik * odds * tracer_2) / deltat;
  }
  src = src1 + src2; // total stratiform or convective scavenging
  fin = src1 / (src + small_value_36); // fraction taken by incloud processes
}
// =============================================================================
// =============================================================================
KOKKOS_INLINE_FUNCTION
Real compute_evap_frac(const Real pdel_ik, const Real evap_ik,
                       const Real precabx) {
  // clang-format off
  //  ------------------------------------------------------------------------------
  //  calculate the fraction of strat precip from above
  //                  which evaporates within this layer
  //  ------------------------------------------------------------------------------
  /*
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: pdel_ik       ! pressure thikness at current column and level [Pa]
  in :: evap_ik       ! evaporation in this layer [kg/kg/s]
  in :: precabx       ! precipitation from above [kg/m2/s]
  out :: fracevx      ! fraction of evaporation [fraction]
  */
  // clang-format on
  // BAD CONSTANT
  constexpr Real small_value_12 = 1.e-12;
  const Real gravit = Constants::gravity;
  Real fracevx =
      evap_ik * pdel_ik / gravit / mam4::max(small_value_12, precabx);
  // trap to ensure reasonable ratio bounds
  fracevx = utils::min_max_bound(0., 1., fracevx);
  return fracevx;
}
// =============================================================================
// =============================================================================
KOKKOS_INLINE_FUNCTION
Real rain_mix_ratio(const Real temperature, const Real pmid,
                    const Real sumppr) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  Purpose:
  //  calculate rain mixing ratio from precipitation rate above.
  //
  //  extracted from clddiag subroutine
  //  for C++ portint, Shuaiqi Tang in 9/22/2022
  // -----------------------------------------------------------------------
  /*
  in :: temperature      ! temperature [K]
  in :: pmid   ! pressure at layer midpoints [Pa]
  in :: sumppr ! sum of precipitation rate above each layer [kg/m2/s]
  out :: rain ! mixing ratio of rain [kg/kg]
  */
  // clang-format on

  // BAD CONSTANT
  constexpr Real small_value_14 = 1.e-14;
  const Real gravit = Constants::gravity;
  const Real rhoh2o = Constants::density_h2o;
  const Real tmelt = Constants::freezing_pt_h2o;
  const Real rair = Constants::r_gas_dry_air;

  // define the constant convfw. taken from cldwat.F90
  // reference: Tripoli and Cotton (1980)
  // falling velocity at air density = 1 kg/m3 [m/s * mam4::sqrt(rho)]
  const Real convfw = 1.94 * 2.13 * mam4::sqrt(rhoh2o * gravit * 2.7e-4);

  Real rain = 0;
  if (temperature > tmelt) {
    // rho =air density [kg/m3]
    const Real rho = pmid / (rair * temperature);
    //  vfall = calculated raindrop falling velocity [m/s]
    const Real vfall = convfw / mam4::sqrt(rho);
    rain = sumppr / (rho * vfall);
    if (rain < small_value_14)
      rain = 0;
  }
  return rain;
}

// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
Real wetdep_resusp(const int is_st_cu, const int mam_prevap_resusp_optcc,
                   const Real pdel_ik, const Real evapx, const Real precabx_old,
                   const Real precabx_base_old, const Real scavabx_old,
                   const Real precnumx_base_old) {
  // clang-format off
  // ------------------------------------------------------------------------------
  // do precip production, resuspension and scavenging
  // ------------------------------------------------------------------------------
  /*
  in :: is_st_cu ! options for stratiform (1) or convective (2) clouds
                      ! raindrop size distribution is
                      ! different for different cloud:
                      ! 1: assume marshall-palmer distribution
                      ! 2: assume log-normal distribution
  in :: mam_prevap_resusp_optcc       ! suspension options
  in :: pdel_ik       ! pressure thikness at current column and level [Pa]
  in :: evapx         ! evaporation at current layer [kg/kg/s]
  in :: precabx_base_old ! input of precipitation at cloud base [kg/m2/s]
  in :: precabx_old ! input of precipitation above this layer [kg/m2/s]
  in :: precnumx_base_old ! input of precipitation number at cloud base [#/m2/s]
  out :: resusp_x    ! aerosol mass re-suspension in a particular layer [kg/m2/s]
  */
  // clang-format on

  // BAD CONSTANT
  constexpr Real small_value_30 = 1.e-30;
  const Real gravit = Constants::gravity;

  Real resusp_x = 0;
  const Real tmpa = mam4::max(0.0, evapx * pdel_ik / gravit);
  const Real precabx_new =
      utils::min_max_bound(0.0, precabx_base_old, precabx_old - tmpa);

  if (precabx_new < small_value_30) {
    // precip rate is essentially zero so do complete resuspension
    resusp_x = wetdep_resusp_noprecip(is_st_cu, mam_prevap_resusp_optcc,
                                      precabx_old, precabx_base_old,
                                      scavabx_old, precnumx_base_old);
  } else if (evapx <= 0.0) {
    // no evap so no resuspension
    resusp_x = 0.0;
  } else {
    // regular non-linear resuspension
    resusp_x = wetdep_resusp_nonlinear(
        is_st_cu, mam_prevap_resusp_optcc, precabx_old, precabx_base_old,
        scavabx_old, precnumx_base_old, precabx_new);
  }
  return resusp_x;
}

// ==============================================================================
// ==============================================================================
KOKKOS_INLINE_FUNCTION
void wetdepa_v2(const Real deltat, const Real pdel, const Real cmfdqr,
                const Real evapc, const Real dlf, const Real conicw,
                const Real precs, const Real evaps, const Real cwat,
                const Real cldt, const Real cldc, const Real cldvcu,
                const Real cldvst, const Real sol_factb, const Real sol_facti,
                const Real sol_factic, const int mam_prevap_resusp_optcc,
                const bool is_strat_cloudborne, const Real scavcoef,
                const Real f_act_conv, const Real tracer, const Real qqcw,
                const Real precabs, const Real precabc, const Real scavabs,
                const Real scavabc, const Real precabs_base,
                const Real precabc_base, const Real precnums_base,
                const Real precnumc_base, Real &scavt, Real &bcscavt,
                Real &rcscavt, Real &rsscavt) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  Purpose:
  //  scavenging code for very soluble aerosols
  //
  //  Author: P. Rasch
  //  Modified by T. Bond 3/2003 to track different removals
  //  Sungsu Park. Mar.2010 : Impose consistencies with a few changes in physics.

  //  this section of code is for highly soluble aerosols,
  //  the assumption is that within the cloud that
  //  all the tracer is in the cloud water
  //
  //  for both convective and stratiform clouds,
  //  the fraction of cloud water converted to precip defines
  //  the amount of tracer which is pulled out.
  // -----------------------------------------------------------------------
  /*
  in ::
         deltat,   ! time step [s]
         pdel,     ! pressure thikness [Pa]
         cmfdqr,   ! rate of production of convective precip [kg/kg/s]
         evapc,    ! Evaporation rate of convective precipitation [kg/kg/s]
         dlf,      ! Detrainment of convective condensate [kg/kg/s]
         conicw,   ! convective cloud water [kg/kg]
         precs,    ! rate of production of stratiform precip [kg/kg/s]
         evaps,    ! rate of evaporation of precip [kg/kg/s]
         cwat,     ! cloud water amount [kg/kg]
         cldt,     ! total cloud fraction [fraction]
         cldc,     ! convective cloud fraction [fraction]
         cldvcu,   ! Convective precipitation area at the top interface of each layer [fraction]
         cldvst,   ! Stratiform precipitation area at the top interface of each layer [fraction]
         tracer    ! trace species [kg/kg]

  in :: mam_prevap_resusp_optcc ! suspension options.
     0 = no resuspension
     1 = linear resuspension of aerosol mass or number following original mam
         coding and history_aero_prevap_resusp = .false.
     2 = same as 1 but history_aero_prevap_resusp = .true.
     3 = same as 2 but with some added "xxx = max( 0, xxx)" lines
   130 = non-linear resuspension of aerosol mass   based on scavenged aerosol mass
   230 = non-linear resuspension of aerosol number based on raindrop number
     (1,2,3 are not used in the current code)


  in :: is_strat_cloudborne = true if tracer is stratiform-cloudborne aerosol; else false

  in :: scavcoef ! Dana and Hales coefficient [1/mm]
  in :: f_act_conv ! [fraction] ! f_act_conv = conv-cloud activation fraction when is_strat_cloudborne==.false.; else 0.0
  in :: qqcw  ! [kg/kg] ! qqcw = strat-cloudborne aerosol corresponding to tracer when is_strat_cloudborne==.false.; else 0.0
  in :: sol_factb   ! solubility factor (frac of aerosol scavenged below cloud) [fraction]
  in :: sol_facti   ! solubility factor (frac of aerosol scavenged in cloud) [fraction]
  in :: sol_factic  ! sol_facti for convective clouds [fraction]

  out :: scavt   ! scavenging tend [kg/kg/s]
  out :: bcscavt  ! below cloud, convective [kg/kg/s]
  out :: rcscavt  ! resuspension, convective [kg/kg/s]
  out :: rsscavt  ! resuspension, stratiform [kg/kg/s]
  */
  // clang-format on
  // BAD CONSTANT
  constexpr Real small_value_2 = 1.e-2;
  constexpr Real small_value_12 = 1.e-12;
  constexpr Real small_value_36 = 1.e-36;

  // omsm = (1 - small number) used to prevent roundoff errors below zero
  // in Fortran EPSILON(X) returns the smallest number E of the same kind
  // as X such that 1 + E > 1.
  // C++ Returns the machine epsilon, that is, the difference between 1.0
  // and the next value representable by the floating-point type T.
  const Real omsm = 1.0 - 2 * epsilon();
  // precabs, precabc, scavabs, scavabc, precabs_base, precabc_base,
  // precnums_base, and precnumc_base are input/output in this routine.
  // precabs: strat precip from above [kg/m2/s]
  // precabc: conv precip from above [kg/m2/s]
  // scavabs: stratiform scavenged tracer flux from above [kg/m2/s]
  // scavabc: convective scavenged tracer flux from above [kg/m2/s]
  // precabs_base: strat precip at an effective cloud base for calculations in a
  // particular layer [kg/m2/s] precabc_base: conv precip at an effective cloud
  // base for calculations in a particular layer [kg/m2/s] precnums_base:
  // stratiform precip number flux at the bottom of a particular layer [#/m2/s]
  // precnumc_base: convective precip number flux at the bottom of a particular
  // layer [#/m2/s]
  // ****************** Evaporation **************************
  // stratiform
  // fraction of stratiform precip from above that is evaporating [fraction]
  const Real fracev_st =
      mam_prevap_resusp_optcc ? compute_evap_frac(pdel, evaps, precabs) : 0;
  // convective
  // Fraction of convective precip from above that is evaporating [fraction]
  const Real fracev_cu =
      mam_prevap_resusp_optcc ? compute_evap_frac(pdel, evapc, precabc) : 0;

  // ****************** Scavenging **************************

  // temporary saved tracer value
  const Real clddiff = cldt - cldc;
  // temporarily calculation of tracer [kg/kg]
  const Real tracer_tmp = mam4::min(
      qqcw, tracer * (clddiff / mam4::max(small_value_2, (1. - clddiff))));
  // calculate in-cumulus and mean tracer values for wetdep_scavenging use
  // in-cumulus tracer concentration [kg/kg]
  const Real tracer_incu = f_act_conv * (tracer + tracer_tmp);
  // mean tracer concenration [kg/kg]
  Real tracer_mean =
      tracer * (1. - cldc * f_act_conv) - cldc * f_act_conv * tracer_tmp;
  tracer_mean = mam4::max(0., tracer_mean);

  // now do the convective scavenging

  // fracp: fraction of convective cloud water converted to rain
  // Sungsu: Below new formula of 'fracp' is necessary since 'conicw'
  // is a LWC/IWC that has already precipitated out, that is, 'conicw' does
  // not contain precipitation at all !
  Real fracp =
      cmfdqr * deltat /
      mam4::max(small_value_12, cldc * conicw + (cmfdqr + dlf) * deltat);
  fracp = utils::min_max_bound(0.0, 1.0, fracp) * cldc;

  Real srcc = 0; // tendency for convective rain scavenging [kg/kg/s]
  Real finc = 0; // fraction of rem. rate by conv. rain [fraction]
  // 2 is for convective:
  wetdep_scavenging(2, is_strat_cloudborne, deltat, fracp, precabc, cldvcu,
                    scavcoef, sol_factb, sol_factic, tracer_incu, tracer_mean,
                    srcc, finc);

  // now do the stratiform scavenging

  // fracp: fraction of convective cloud water converted to rain
  fracp = precs * deltat / mam4::max(small_value_12, cwat + precs * deltat);
  fracp = utils::min_max_bound(0.0, 1.0, fracp);

  Real srcs = 0; // tendency for stratiform rain scavenging [kg/kg/s]
  Real fins = 0; // fraction of rem. rate by strat rain [fraction]
  // 1 for stratiform:
  wetdep_scavenging(1, is_strat_cloudborne, deltat, fracp, precabs, cldvst,
                    scavcoef, sol_factb, sol_facti, tracer, tracer_mean, srcs,
                    fins);

  // rat =  ratio of amount available to amount removed [fraction]
  // make sure we dont take out more than is there
  // ratio of amount available to amount removed
  const Real rat = tracer / mam4::max(deltat * (srcc + srcs), small_value_36);
  if (rat < 1) {
    srcs = srcs * rat;
    srcc = srcc * rat;
  }
  // total scavenging tendency [kg/kg/s]
  const Real srct = (srcc + srcs) * omsm;
  // ****************** Resuspension **************************

  // aerosol mass re-suspension in a particular layer from convective rain
  // [kg/m2/s]
  Real resusp_c = 0;
  // aerosol mass re-suspension in a particular layer from stratiform rain
  // [kg/m2/s]
  Real resusp_s = 0;
  // tend is all tracer removed by scavenging, plus all re-appearing from
  // evaporation above
  if (mam_prevap_resusp_optcc >= 100) {
    // step 1 - do evaporation and resuspension
    resusp_s = wetdep_resusp(1, mam_prevap_resusp_optcc, pdel, evaps, precabs,
                             precabs_base, scavabs, precnums_base);

    // for convective clouds
    resusp_c = wetdep_resusp(2, mam_prevap_resusp_optcc, pdel, evapc, precabc,
                             precabc_base, scavabc, precnumc_base);
  } else { // mam_prevap_resusp_optcc = 0, no resuspension
    resusp_c = fracev_cu * scavabc;
    resusp_s = fracev_st * scavabs;
  }

  // ****************** update scavengingfor output ***************
  scavt = 0.0; // scavenging tend at current  [kg/kg/s]
  bcscavt =
      0.0; // below cloud, convective scavenging tends at current [kg/kg/s]
  rcscavt = 0.0; // resuspension, convective tends at current  [kg/kg/s]
  rsscavt = 0.0; // resuspension, stratiform tends at current  [kg/kg/s]

  const Real gravit = Constants::gravity;

  if (mam_prevap_resusp_optcc == 0)
    scavt = -srct + (fracev_st * scavabs + fracev_cu * scavabc) * gravit / pdel;
  else
    scavt = -srct + (resusp_s + resusp_c) * gravit / pdel;

  if (mam_prevap_resusp_optcc == 0) {
    bcscavt = -(srcc * (1 - finc)) * omsm + fracev_cu * scavabc * gravit / pdel;
    rcscavt = 0.0;
    rsscavt = 0.0;
  } else {
    bcscavt = -(srcc * (1 - finc)) * omsm;
    rcscavt = resusp_c * gravit / pdel;
    rsscavt = resusp_s * gravit / pdel;
  }
}
// ==============================================================================

/**
 * @brief Estimate the cloudy volume which is occupied by rain or cloud water as
 *        the max between the local cloud amount or the sum above of
 *       (cloud * positive precip production)   sum total precip from above
 *        ----------------------------------- X -------------------------
 *        sum above of ( positive precip )      sum positive precip from above
 *
 * @param[in] temperature Temperature [K].
 * @param[in] pmid Pressure at layer midpoints [Pa].
 * @param[in] pdel Pressure difference across layers [Pa].
 * @param[in] cmfdqr to convective rainout [kg/kg/s].
 * @param[in] evapc Evaporation rate of convective precipitation ( >= 0 )
 * [kg/kg/s].
 * @param[in] cldt Total cloud fraction [fraction, unitless].
 * @param[in] cldcu Cumulus cloud fraction [fraction, unitless].
 * @param[in] clst Stratus cloud fraction [fraction, unitless].
 * @param[in] evapr rate of evaporation of falling precipitation [kg/kg/s].
 * @param[in] prain rate of conversion of condensate to precipitation [kg/kg/s].
 * @param[in] atm Atmosphere object (used for number of levels).
 *
 * @param[out] cldv Fraction occupied by rain or cloud water [fraction,
 * unitless].
 * @param[out] cldvcu Convective precipitation volume [fraction, unitless].
 * @param[out] cldvst Stratiform precipitation volume [fraction, unitless].
 * @param[out] rain Rain mixing ratio [kg/kg].
 *
 * @pre In F90, ncol == 1 as we only operate over one column at a time
 *      as outer loops will iterate over columns, so we drop ncol as input.
 * @pre In F90 code, pcols is the number of columns in the mesh.
 *      Since we are only operating over one column, pcols == 1.
 * @pre In F90 code, ncol == pcols. Since ncol == 1, pcols == 1.
 *
 * @pre atm is initialized correctly and has the correct number of levels.
 */
KOKKOS_INLINE_FUNCTION
void clddiag(const int nlev, const Real *temperature, const Real *pmid,
             const Real *pdel, const Real *cmfdqr, const Real *evapc,
             const Real *cldt, const Real *cldcu, const Real *cldst,
             const Real *evapr, const Real *prain, Real *cldv, Real *cldvcu,
             Real *cldvst, Real *rain) {
  // Calculate local precipitation production rate
  // In src/chemistry/aerosol/wetdep.F90, (prain + cmfdqr) is used for
  // source_term

  // FIXME: Possible bug: why there is no evapc in sumppr_all calculation?
  Real sumppr_all = 0.0;
  Real g = Constants::gravity;
  // NOTE: we can't use a parallel_scan here because rain[i] is updated using
  // the partially NOTE: summed sumppr_all
  for (int i = 0; i < nlev; i++) {
    const Real source_term = prain[i] + cmfdqr[i];
    Real lprec = local_precip_production(pdel[i], source_term, evapr[i], g);
    sumppr_all += lprec;
    // Calculate rain mixing ratio
    rain[i] = rain_mix_ratio(temperature[i], pmid[i], sumppr_all);
  }

  // Calculate cloudy volume which is occupied by rain or cloud water
  // Total
  auto prec = [&](int i) -> Real {
    const Real source_term = prain[i] + cmfdqr[i];
    Real lprec = local_precip_production(pdel[i], source_term, evapr[i], g);
    return lprec;
  };
  calculate_cloudy_volume(nlev, cldt, prec, true, cldv);

  // Convective
  auto prec_cu = [&](int i) -> Real {
    Real lprec = local_precip_production(pdel[i], cmfdqr[i], evapc[i], g);
    return lprec;
  };
  calculate_cloudy_volume(nlev, cldcu, prec_cu, false, cldvcu);

  // Stratiform
  auto prec_st = [&](int i) -> Real {
    Real lprec = local_precip_production(pdel[i], prain[i], evapr[i], g);
    return lprec;
  };
  calculate_cloudy_volume(nlev, cldst, prec_st, false, cldvst);
}

template <typename VIEWTYPE>
KOKKOS_INLINE_FUNCTION void sum_values(const ThreadTeam &team,
                                       const View1D &sum, VIEWTYPE x,
                                       VIEWTYPE y, const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev),
                       [&](int k) { sum[k] = x[k] + y[k]; });
}
KOKKOS_INLINE_FUNCTION
void zero_values(const ThreadTeam &team, const View1D &vec, const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev),
                       [&](int k) { vec[k] = 0.0; });
}

KOKKOS_INLINE_FUNCTION
void sum_deep_and_shallow(const ThreadTeam &team, const View1D &conicw,
                          const ConstView1D &icwmrdp,
                          const ConstView1D &dp_frac,
                          const ConstView1D &icwmrsh,
                          const ConstView1D &sh_frac, const int nlev) {
  // BAD CONSTANT
  constexpr Real small_value_2 = 1.e-2;
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    // sum deep and shallow convection contributions
    conicw[k] = (icwmrdp[k] * dp_frac[k] + icwmrsh[k] * sh_frac[k]) /
                mam4::max(small_value_2, sh_frac[k] + dp_frac[k]);
  });
}

KOKKOS_INLINE_FUNCTION
void cloud_diagnostics(const ThreadTeam &team, ConstColumnView temperature,
                       ConstColumnView pmid, ConstColumnView pdel,
                       const View1D &cmfdqr, const View1D &evapc,
                       const ConstColumnView &cldt, const View1D &cldcu,
                       const View1D &cldst, const ConstColumnView &evapr,
                       const ConstColumnView &prain,
                       // outputs
                       const View1D &cldv, const View1D &cldvcu,
                       const View1D &cldvst, const View1D &rain,
                       const int nlev) {
  // NOTE: The k loop inside clddiag cannot be converted to parallel_for
  // because precabs requires values from the previous elevation (k-1).
  Kokkos::single(Kokkos::PerTeam(team), [=]() {
    wetdep::clddiag(nlev, temperature.data(), pmid.data(), pdel.data(),
                    cmfdqr.data(), evapc.data(), cldt.data(), cldcu.data(),
                    cldst.data(), evapr.data(), prain.data(),
                    // outputs
                    cldv.data(), cldvcu.data(), cldvst.data(), rain.data());
  });
}

KOKKOS_INLINE_FUNCTION
void set_f_act(const ThreadTeam &team, int *isprx,
               const View1D &f_act_conv_coarse,
               const View1D &f_act_conv_coarse_dust,
               const View1D &f_act_conv_coarse_nacl, ConstColumnView pdel,
               ConstColumnView prain, const View1D &cmfdqr,
               const ConstView1D &evapr, const View2D &state_q,
               const View2D &ptend_q, const Real dt, const int nlev) {

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    isprx[k] = aero_model::examine_prec_exist(k, pdel.data(), prain.data(),
                                              cmfdqr.data(), evapr.data());

    aero_model::set_f_act_coarse(k, state_q, ptend_q, dt, f_act_conv_coarse[k],
                                 f_act_conv_coarse_dust[k],
                                 f_act_conv_coarse_nacl[k]);
  });
}

// Computes lookup table for aerosol impaction/interception scavenging rates
KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_get(const ThreadTeam &team, const Diagnostics &diags,
                               const int *isprx, const View2D &scavimptblvol,
                               const View2D &scavimptblnum,
                               const View1D &scavcoefnum,
                               const View1D &scavcoefvol, const int imode,
                               const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    scavcoefnum[k] = scavcoefvol[k] = 0.0;
    const bool let_it_rain = (isprx[k] == 1);
    if (let_it_rain) {
      const Real dgnum_amode_imode = modes(imode).nom_diameter;
      ColumnView dgn_awet_imode = diags.wet_geometric_mean_diameter_i[imode];
      const Real dgn_awet_imode_k = dgn_awet_imode[k];
      aero_model::modal_aero_bcscavcoef_get(
          imode, dgn_awet_imode_k, dgnum_amode_imode, scavimptblvol,
          scavimptblnum, scavcoefnum[k], scavcoefvol[k]);
    }
  });
}

// Computes lookup table for aerosol impaction/interception scavenging rates
KOKKOS_INLINE_FUNCTION
void modal_aero_bcscavcoef_get(const ThreadTeam &team,
                               const View2D &wet_geometric_mean_diameter_i,
                               const int *isprx, const View2D &scavimptblvol,
                               const View2D &scavimptblnum,
                               const View1D &scavcoefnum,
                               const View1D &scavcoefvol, const int imode,
                               const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    scavcoefnum[k] = scavcoefvol[k] = 0.0;
    const bool let_it_rain = (isprx[k] == 1);
    if (let_it_rain) {
      const Real dgnum_amode_imode = modes(imode).nom_diameter;
      const Real dgn_awet_imode_k = wet_geometric_mean_diameter_i(imode, k);
      aero_model::modal_aero_bcscavcoef_get(
          imode, dgn_awet_imode_k, dgnum_amode_imode, scavimptblvol,
          scavimptblnum, scavcoefnum[k], scavcoefvol[k]);
    }
  });
}

// define sol_factb and sol_facti values, and f_act_conv
KOKKOS_INLINE_FUNCTION
void define_act_frac(const ThreadTeam &team, const View1D &sol_facti,
                     const View1D &sol_factic, const View1D &sol_factb,
                     const View1D &f_act_conv, const int lphase,
                     const int imode, const int nlev,
                     const Real scav_fraction_in_cloud_strat,
                     const Real scav_fraction_in_cloud_conv,
                     const Real scav_fraction_below_cloud_strat,
                     const Real activation_fraction_in_cloud_conv) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int k) {
    aero_model::define_act_frac(lphase, imode, scav_fraction_in_cloud_strat,
                                scav_fraction_in_cloud_conv,
                                scav_fraction_below_cloud_strat,
                                activation_fraction_in_cloud_conv, sol_facti[k],
                                sol_factic[k], sol_factb[k], f_act_conv[k]);
  });
}

KOKKOS_INLINE_FUNCTION
void compute_q_tendencies(
    const ThreadTeam &team, const int nlev,
    // const Prognostics &progs,
    const View1D &f_act_conv, const View1D &f_act_conv_coarse,
    const View1D &f_act_conv_coarse_dust, const View1D &f_act_conv_coarse_nacl,
    const View1D &scavcoefnum, const View1D &scavcoefvol, const View1D &totcond,
    const View1D &cmfdqr, const View1D &conicw, const View1D &evapc,
    const ConstView1D &evapr, const ConstView1D &prain, const ConstView1D &dlf,
    const ConstView1D &cldt, const View1D &cldcu, const View1D &cldst,
    const View1D &cldvst, const View1D &cldvcu, const View1D &sol_facti,
    const View1D &sol_factic, const View1D &sol_factb, const View1D &scavt,
    const View1D &bcscavt, const View1D &rcscavt, const View2D &rtscavt_sv,
    const View2D &state_q, const View2D &qqcw, const View2D &ptend_q,
    // Kokkos::View<Real * [aero_model::maxd_aspectype + 2][aero_model::pcnst]>
    //     qqcw_sav,
    ConstColumnView pdel, const Real dt, const int jnummaswtr, const int jnv,
    const int mm, const int lphase, const int imode, const int lspec,
    View2D workspace) {

  team.team_barrier();
  // clang-format off
  //   0 = no resuspension
  // 130 = non-linear resuspension of aerosol mass based on scavenged aerosol mass
  // 230 = non-linear resuspension of aerosol number based on raindrop number
  // the 130 thru 230 all use the new prevap_resusp code block in subr wetdepa_v2
  // clang-format on
  const int mam_prevap_resusp_no = 0;
  const int mam_prevap_resusp_mass = 130;
  const int mam_prevap_resusp_num = 230;
  const int jaeronumb = 0;
  const int jaeromass = 1;
  const int modeptr_coarse = static_cast<int>(ModeIndex::Coarse);

  // clang-format off
  // mam_prevap_resusp_optcc values control the prevap_resusp
  // calculations in wetdepa_v2:
  //     0 = no resuspension
  //   130 = non-linear resuspension of aerosol mass based on
  //         scavenged aerosol mass 
  //   230 = non-linear resuspension of aerosol number based on
  //         raindrop number the 130 thru 230 all use the new
  //         prevap_resusp code block in subr wetdepa_v2
  // clang-format on
  int optcc = mam_prevap_resusp_no;
  {
    if (jnummaswtr == jaeromass)
      // dry mass
      optcc = mam_prevap_resusp_mass;
    else if (jnummaswtr == jaeronumb && lphase == 1 && imode == modeptr_coarse)
      // number
      optcc = mam_prevap_resusp_num;
  }
  const int mam_prevap_resusp_optcc = optcc;

  // set f_act_conv for interstitial (lphase=1) coarse mode
  // species for the convective in-cloud, we conceptually treat
  // the coarse dust and seasalt as being externally mixed, and
  // apply f_act_conv = f_act_conv_coarse_dust/nacl to
  // dust/seasalt number and sulfate are conceptually partitioned
  // to the dust and seasalt on a mass basis, so the f_act_conv
  // for number and sulfate are mass-weighted averages of the
  // values used for dust/seasalt
  if (lphase == 1 && imode == modeptr_coarse) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                         [&](int k) { f_act_conv[k] = f_act_conv_coarse[k]; });
    if (jnummaswtr == jaeromass) {
      if (aero_model::lmassptr_amode(lspec, imode) ==
          aero_model::lptr_dust_a_amode(imode)) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
          f_act_conv[k] = f_act_conv_coarse_dust[k];
        });
      } else if (aero_model::lmassptr_amode(lspec, imode) ==
                 aero_model::lptr_nacl_a_amode(imode)) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
          f_act_conv[k] = f_act_conv_coarse_nacl[k];
        });
      }
    }
  }

  // currently compute_q_tendencies_worksparce[14]
  View1D evap = ekat::subview(workspace, 0);
  View1D rain = ekat::subview(workspace, 1);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
    const Real gravit = Constants::gravity;
    evap[k] = mam4::max(0.0, evapr[k] * pdel[k] / gravit);
    rain[k] = mam4::max(0.0, prain[k] * pdel[k] / gravit);
  });

  // (look below for workspace 2)
  View1D precabs = ekat::subview(workspace, 3);
  View1D precabs_tmp = ekat::subview(workspace, 4);
  View1D precabs_base = ekat::subview(workspace, 5);
  View1D precabs_base_tmp = ekat::subview(workspace, 6);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
    precabs[k] = 0.0;
    precabs_tmp[k] = 0.0;
    precabs_base[k] = 0.0;
    precabs_base_tmp[k] = 0.0;
  });

  team.team_barrier();
  if (mam_prevap_resusp_optcc >= 100) {
    View1D bndd = ekat::subview(workspace, 2);
    Kokkos::single(Kokkos::PerTeam(team), [=]() {
      // Because of these two values, the loop can not be parallel_for
      Real prec = 0.0, prec_base = 0.0;
      constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
      for (int k = 0; k < nlev - 1; ++k) {
        bndd[k] = utils::min_max_bound(0.0, prec_base, prec - evap[k]);
        precabs_base_tmp[k] = prec_base;
        if (bndd[k] < small_value_30) {
          // setting both these precip rates to zero causes the resuspension
          // calculations to start fresh if there is any more precip production
          precabs_base_tmp[k] = 0.0;
          bndd[k] = 0.0;
        }
        prec_base = mam4::max(0.0, precabs_base_tmp[k] + rain[k]);
        prec = utils::min_max_bound(0.0, prec_base, bndd[k] + rain[k]);
      }
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
      precabs_base[k + 1] = mam4::max(0.0, precabs_base_tmp[k] + rain[k]);
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
      precabs[k + 1] =
          utils::min_max_bound(0.0, precabs_base[k + 1], bndd[k] + rain[k]);
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
      precabs_tmp[k] =
          utils::min_max_bound(0.0, precabs_base[k], precabs[k] - evap[k]);
    });
  } else if (mam_prevap_resusp_optcc == 0) {
    View1D netc = ekat::subview(workspace, 2);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
      const Real gravit = Constants::gravity;
      netc[k] = (prain[k] - evapr[k]) * pdel[k] / gravit;
    });
    team.team_barrier();
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, nlev),
                          [&](const int k, Real &accumulator, const bool last) {
                            if (last)
                              precabs[k] = accumulator;
                            accumulator += netc[k];
                          });
  }

  View1D precnums_base = ekat::subview(workspace, 7);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                       [&](int k) { precnums_base[k] = 0.0; });
  // Yes, it is redundant but keys off of the old code.
  if (mam_prevap_resusp_optcc >= 100 && mam_prevap_resusp_optcc > 130) {
    constexpr Real small_value_2 = 1.e-2;   // BAD CONSTANT
    constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
    View1D copy_from_prev = ekat::subview(workspace, 2);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                         [&](int k) { copy_from_prev[k] = 0.0; });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
      // raindrop number increase
      if (precabs_base[k + 1] < small_value_30) {
        precnums_base[k + 1] = 0.0;
      } else if (precabs_base[k + 1] > precabs_base_tmp[k]) {
        // note - calc rainshaft number flux from rainshaft water flux,
        // then multiply by rainshaft area to get grid-average number flux
        const Real arainx = mam4::max(cldvst[k + 1], small_value_2);
        const Real tmpa = arainx * flux_precnum_vs_flux_prec_mpln(
                                       precabs_base[k + 1] / arainx, 1);
        precnums_base[k + 1] = mam4::max(0.0, tmpa);
      } else {
        copy_from_prev[k + 1] = 1;
      }
    });
    team.team_barrier();
    // The last else above statement can not be done in parallel
    Kokkos::single(Kokkos::PerTeam(team), [=]() {
      for (int k = 0; k < nlev - 1; ++k) {
        if (copy_from_prev[k + 1])
          precnums_base[k + 1] = precnums_base[k];
      }
    });
  }

  View1D cmfd = ekat::subview(workspace, 1);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
    const Real gravit = Constants::gravity;
    evap[k] = mam4::max(0.0, evapc[k] * pdel[k] / gravit);
    cmfd[k] = mam4::max(0.0, cmfdqr[k] * pdel[k] / gravit);
  });

  View1D precabc = ekat::subview(workspace, 8);
  View1D precabc_base = ekat::subview(workspace, 10);
  View1D precnumc_base = ekat::subview(workspace, 11);
  View1D precabc_base_tmp = ekat::subview(workspace, 6);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
    precabc[k] = 0.0;
    precabc_base[k] = 0.0;
    precnumc_base[k] = 0.0;
    precabc_base_tmp[k] = 0.0;
  });

  if (mam_prevap_resusp_optcc >= 100) {
    View1D bndd = ekat::subview(workspace, 2);
    Kokkos::single(Kokkos::PerTeam(team), [=]() {
      Real prec = 0, prec_base = 0.0;
      constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
      for (int k = 0; k < nlev - 1; ++k) {
        bndd[k] = utils::min_max_bound(0.0, prec_base, prec - evap[k]);
        precabc_base_tmp[k] = prec_base;
        if (bndd[k] < small_value_30) {
          // setting both these precip rates to zero causes the resuspension
          // calculations to start fresh if there is any more precip production
          precabc_base_tmp[k] = 0.0;
          bndd[k] = 0.0;
        }
        prec_base = mam4::max(0.0, precabc_base_tmp[k] + cmfd[k]);
        prec = utils::min_max_bound(0.0, prec_base, bndd[k] + cmfd[k]);
      }
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
      precabc_base[k + 1] = mam4::max(0.0, precabc_base_tmp[k] + cmfd[k]);
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
      precabc[k + 1] =
          utils::min_max_bound(0.0, precabc_base[k + 1], bndd[k] + cmfd[k]);
    });
  } else if (mam_prevap_resusp_optcc == 0) {
    View1D netc = ekat::subview(workspace, 2);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
      const Real gravit = Constants::gravity;
      netc[k] = (cmfdqr[k] - evapc[k]) * pdel[k] / gravit;
    });
    team.team_barrier();
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, nlev),
                          [&](const int k, Real &accumulator, const bool last) {
                            if (last)
                              precabc[k] = accumulator;
                            accumulator += netc[k];
                          });
  }

  // Yes, it is redundant but keys off of the old code.
  if (mam_prevap_resusp_optcc >= 100 && mam_prevap_resusp_optcc > 130) {
    // This can be done as three parallel loops, push the loop into the
    // conditional
    constexpr Real small_value_2 = 1.e-2;   // BAD CONSTANT
    constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
    View1D copy_from_prev = ekat::subview(workspace, 2);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                         [&](int k) { copy_from_prev[k] = 0.0; });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
      // raindrop number increase
      if (precabc_base[k + 1] < small_value_30) {
        precnumc_base[k + 1] = 0.0;
      } else if (precabc_base[k + 1] > precabc_base_tmp[k]) {
        // note - calc rainshaft number flux from rainshaft water flux,
        // then multiply by rainshaft area to get grid-average number flux
        const Real arainx = mam4::max(cldvst[k + 1], small_value_2);
        const Real evap = arainx * flux_precnum_vs_flux_prec_mpln(
                                       precabc_base[k + 1] / arainx, 2);
        precnumc_base[k + 1] = mam4::max(0.0, evap);
      } else {
        copy_from_prev[k + 1] = 1;
      }
    });
    team.team_barrier();
    // The last else above statement can not be done in parallel
    Kokkos::single(Kokkos::PerTeam(team), [=]() {
      for (int k = 0; k < nlev - 1; ++k) {
        if (copy_from_prev[k + 1])
          precnumc_base[k + 1] = precnumc_base[k];
      }
    });
  }
  team.team_barrier();

  View1D srcc = ekat::subview(workspace, 2);
  View1D srcs = ekat::subview(workspace, 0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                       [&](int k) { srcc[k] = 0.0; });
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                       [&](int k) { srcs[k] = 0.0; });
  team.team_barrier();
  const bool is_strat_cloudborne = lphase == 2 ? true : false;
  if (is_strat_cloudborne) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
      // convective cloud does not affect strat-cloudborne aerosol
      const Real src1 = 0.0;
      // no below-cloud scavenging for strat-cloudborne aerosol
      const Real src2 = 0.0;
      // total stratiform or convective scavenging
      srcc[k] = src1 + src2;
    });
  } else {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
      constexpr Real small_value_12 = 1.e-12; // BAD CONSTANT
      constexpr Real small_value_5 = 1.e-5;   // BAD CONSTANT
      constexpr Real small_value_2 = 1.e-2;   // BAD CONSTANT
      const Real tqqcw = qqcw(k, mm);
      const Real tracer = state_q(k, mm) + ptend_q(k, mm) * dt;

      Real scavcoef = 0.0;
      if (jnv)
        scavcoef = (1 == jnv) ? scavcoefnum[k] : scavcoefvol[k];

      Real fracp = cmfdqr[k] * dt /
                   mam4::max(small_value_12,
                             cldcu[k] * conicw[k] + (cmfdqr[k] + dlf[k]) * dt);
      fracp = utils::min_max_bound(0.0, 1.0, fracp) * cldcu[k];
      // temporary saved tracer value
      const Real clddiff = cldt[k] - cldcu[k];
      // temporarily calculation of tracer [kg/kg]
      const Real tracer_tmp = mam4::min(
          tqqcw, tracer * (clddiff / mam4::max(small_value_2, (1. - clddiff))));
      // calculate in-cumulus and mean tracer values for wetdep_scavenging use
      // in-cumulus tracer concentration [kg/kg]
      const Real tracer_incu = f_act_conv[k] * (tracer + tracer_tmp);
      // mean tracer concenration [kg/kg]
      Real tracer_mean = tracer * (1. - cldcu[k] * f_act_conv[k]) -
                         cldcu[k] * f_act_conv[k] * tracer_tmp;
      tracer_mean = mam4::max(0., tracer_mean);

      // calculate limitation of removal rate using Dana and Hales coefficient
      // odds  : limit on removal rate (proportional to prec) [fraction]
      Real odds =
          precabc[k] / mam4::max(cldvcu[k], small_value_5) * scavcoef * dt;
      odds = utils::min_max_bound(0.0, 1.0, odds);

      // incloud scavenging tendency [kg/kg/s]
      const Real src1 = (sol_factic[k] * fracp * tracer_incu) / dt;
      // below-cloud scavenging tendency [kg/kg/s]
      const Real src2 = (sol_factb[k] * cldvcu[k] * odds * tracer_mean) / dt;
      // total stratiform or convective scavenging
      srcc[k] = src1 + src2;
    });
  }

  if (is_strat_cloudborne) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
      constexpr Real small_value_12 = 1.e-12; // BAD CONSTANT
      const Real tracer = qqcw(k, mm);
      // strat in-cloud removal only affects strat-cloudborne aerosol
      Real fracp =
          prain[k] * dt / mam4::max(totcond[k] + prain[k] * dt, small_value_12);
      fracp = utils::min_max_bound(0.0, 1.0, fracp);
      // in-cloud scavenging:
      // incloud scavenging tendency [kg/kg/s]
      const Real src1 = (sol_facti[k] * fracp * tracer) / dt;
      // no below-cloud scavenging for strat-cloudborne aerosol
      // total stratiform or convective scavenging
      srcs[k] = src1;
    });
  } else {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
      constexpr Real small_value_5 = 1.e-5; // BAD CONSTANT
      constexpr Real small_value_2 = 1.e-2; // BAD CONSTANT
      const Real tracer = state_q(k, mm) + ptend_q(k, mm) * dt;
      const Real tqqcw = qqcw(k, mm);
      Real scavcoef = 0.0;
      if (jnv)
        scavcoef = (1 == jnv) ? scavcoefnum[k] : scavcoefvol[k];
      // temporary saved tracer value
      const Real clddiff = cldt[k] - cldcu[k];
      // temporarily calculation of tracer [kg/kg]
      const Real tracer_tmp = mam4::min(
          tqqcw, tracer * (clddiff / mam4::max(small_value_2, (1. - clddiff))));
      // mean tracer concenration [kg/kg]
      Real tracer_mean = tracer * (1. - cldcu[k] * f_act_conv[k]) -
                         cldcu[k] * f_act_conv[k] * tracer_tmp;

      // calculate limitation of removal rate using Dana and Hales coefficient
      // odds  : limit on removal rate (proportional to prec) [fraction]
      Real odds =
          precabs[k] / mam4::max(cldvst[k], small_value_5) * scavcoef * dt;
      odds = utils::min_max_bound(0.0, 1.0, odds);
      // strat in-cloud removal only affects strat-cloudborne aerosol
      const Real src2 = (sol_factb[k] * cldvst[k] * odds * tracer_mean) / dt;
      srcs[k] = src2;
    });
  }

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
    constexpr Real small_value_36 = 1.e-36; // BAD CONSTANT
    const Real tracer =
        lphase == 1 ? state_q(k, mm) + ptend_q(k, mm) * dt : qqcw(k, mm);
    const Real rat =
        tracer / mam4::max(dt * (srcc[k] + srcs[k]), small_value_36);
    if (rat < 1) {
      srcs[k] *= rat;
      srcc[k] *= rat;
    }
  });

  team.team_barrier();
  View1D scavabs = ekat::subview(workspace, 12);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                       [&](int k) { scavabs[k] = 0.0; });
  team.team_barrier();
  if (mam_prevap_resusp_optcc >= 100) {
    if (mam_prevap_resusp_optcc <= 130) {
      View1D x_ratio = ekat::subview(workspace, 9);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                           [&](int k) { x_ratio[k] = 0.0; });
      team.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
        constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
        // update aerosol resuspension
        // fraction of precabx and precabx_base
        if (precabs_tmp[k] < small_value_30) {
        } else if (evapr[k] <= 0.0) {
        } else {
          const int is_st_cu = 1;
          const Real u_old =
              utils::min_max_bound(0.0, 1.0, precabs[k] / precabs_base[k]);
          // fraction after calling function *_resusp_vs_fprec_evap_mpln
          // non-linear resuspension of aerosol mass
          Real x_old =
              1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
          x_old = utils::min_max_bound(0.0, 1.0, x_old);
          if (small_value_30 <= x_old) {
            // fraction of precabx and precabx_base
            Real u_new = utils::min_max_bound(0.0, 1.0,
                                              precabs_tmp[k] / precabs_base[k]);
            u_new = mam4::min(u_new, u_old);
            // non-linear resuspension of aerosol mass
            Real x_new =
                1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_new, is_st_cu);
            x_new = utils::min_max_bound(0.0, 1.0, x_new);
            x_new = mam4::min(x_new, x_old);
            x_ratio[k] = utils::min_max_bound(0.0, 1.0, x_new / x_old);
          }
        }
      });
      View1D tmpa = ekat::subview(workspace, 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
        const Real gravit = Constants::gravity;
        tmpa[k] = mam4::max(0.0, srcs[k] * pdel[k] / gravit);
      });
      team.team_barrier();
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int k = 0; k < nlev - 1; ++k) {
          constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
          Real scavabs_tmp = 0.0;
          if (precabs_tmp[k] < small_value_30) {
            scavabs_tmp = 0.0;
          } else if (evapr[k] <= 0.0) {
            // no evap so no resuspension
            scavabs_tmp = scavabs[k];
          } else {
            // aerosol mass resuspension
            scavabs_tmp = mam4::max(0.0, scavabs[k] * x_ratio[k]);
          }
          scavabs_tmp = mam4::max(0.0, scavabs_tmp + tmpa[k]);
          scavabs[k + 1] = scavabs_tmp;
        }
      });
    } else {
      View1D copy_from_prev = ekat::subview(workspace, 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                           [&](int k) { copy_from_prev[k] = 0.0; });
      team.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
        constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
        if (precabs_tmp[k] < small_value_30) {
          copy_from_prev[k + 1] = 1;
        } else if (evapr[k] <= 0.0) {
          copy_from_prev[k + 1] = 1;
        } else {
          // number resuspension
          scavabs[k + 1] = 0.0;
        }
      });
      team.team_barrier();
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int k = 0; k < nlev - 1; ++k) {
          if (copy_from_prev[k + 1])
            scavabs[k + 1] = scavabs[k];
        }
      });
    }
  } else if (mam_prevap_resusp_optcc == 0) {
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, nlev),
                          [&](const int k, Real &accumulator, const bool last) {
                            const Real gravit = Constants::gravity;
                            if (last)
                              scavabs[k] = accumulator;
                            accumulator += srcs[k] * pdel[k] / gravit;
                          });
  }

  View1D scavabc = ekat::subview(workspace, 13);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                       [&](int k) { scavabc[k] = 0.0; });
  team.team_barrier();
  if (mam_prevap_resusp_optcc >= 100) {
    if (mam_prevap_resusp_optcc <= 130) {
      View1D x_ratio = ekat::subview(workspace, 9);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                           [&](int k) { x_ratio[k] = 0.0; });
      team.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
        constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
        if (precabc[k] < small_value_30) {
        } else if (evapr[k] <= 0.0) {
        } else {
          const int is_st_cu = 2;
          // update aerosol resuspension
          // fraction of precabx and precabx_base
          const Real u_old =
              utils::min_max_bound(0.0, 1.0, precabc[k] / precabc_base[k]);
          // fraction after calling function *_resusp_vs_fprec_evap_mpln
          // non-linear resuspension of aerosol mass
          Real x_old =
              1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_old, is_st_cu);
          x_old = utils::min_max_bound(0.0, 1.0, x_old);
          if (small_value_30 <= x_old) {
            // fraction of precabx and precabx_base
            Real u_new =
                utils::min_max_bound(0.0, 1.0, precabc[k] / precabc_base[k]);
            u_new = mam4::min(u_new, u_old);
            // non-linear resuspension of aerosol mass
            Real x_new =
                1.0 - faer_resusp_vs_fprec_evap_mpln(1.0 - u_new, is_st_cu);
            x_new = utils::min_max_bound(0.0, 1.0, x_new);
            x_new = mam4::min(x_new, x_old);
            x_ratio[k] = utils::min_max_bound(0.0, 1.0, x_new / x_old);
          }
        }
      });
      View1D tmpc = ekat::subview(workspace, 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
        const Real gravit = Constants::gravity;
        tmpc[k] = mam4::max(0.0, srcc[k] * pdel[k] / gravit);
      });
      team.team_barrier();
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int k = 0; k < nlev - 1; ++k) {
          constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
          Real scavabc_tmp = 0.0;
          if (precabc[k] < small_value_30) {
            scavabc_tmp = 0.0;
          } else if (evapr[k] <= 0.0) {
            // no evap so no resuspension
            scavabc_tmp = scavabc[k];
          } else {
            // aerosol mass resuspension
            scavabc_tmp = mam4::max(0.0, scavabc[k] * x_ratio[k]);
          }
          scavabc_tmp = mam4::max(0.0, scavabc_tmp + tmpc[k]);
          scavabc[k + 1] = scavabc_tmp;
        }
      });
    } else { // 130 < mam_prevap_resusp_optcc
      View1D copy_from_prev = ekat::subview(workspace, 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev),
                           [&](int k) { copy_from_prev[k] = 0.0; });
      team.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev - 1), [&](int k) {
        constexpr Real small_value_30 = 1.e-30; // BAD CONSTANT
        if (precabc[k] < small_value_30) {
          copy_from_prev[k + 1] = 1;
        } else if (evapr[k] <= 0.0) {
          copy_from_prev[k + 1] = 1;
        } else {
          scavabc[k + 1] = 0.0;
        }
      });
      team.team_barrier();
      Kokkos::single(Kokkos::PerTeam(team), [=]() {
        for (int k = 0; k < nlev - 1; ++k) {
          if (copy_from_prev[k + 1])
            scavabc[k + 1] = scavabc[k];
        }
      });
    }
  } else if (mam_prevap_resusp_optcc == 0) {
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, nlev),
                          [&](const int k, Real &accumulator, const bool last) {
                            const Real gravit = Constants::gravity;
                            if (last)
                              scavabc[k] = accumulator;
                            accumulator += srcc[k] * pdel[k] / gravit;
                          });
  }

  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int k) {
    const auto rtscavt_sv_k = ekat::subview(rtscavt_sv, k);

    // OK, this is from the old mam4: Phase 2 is before Phase 1.
    // Note that the phase loops goes from 2 to 1 in reverse order
    // and the qqcw_sav is set first in phase 2 the used in phase 1.
    if (lphase == 1) {
      // traces reflects changes from modal_aero_calcsize and is the
      // "most current" q

      const Real tracer = state_q(k, mm) + ptend_q(k, mm) * dt;
      const Real qqcw_tmp = qqcw(k, mm);
      Real scavcoef = 0.0;
      if (jnv)
        scavcoef = (1 == jnv) ? scavcoefnum[k] : scavcoefvol[k];

      // resuspension, stratiform [kg/kg/s]
      Real rsscavt = 0.0;
      // is_strat_cloudborne = true if tracer is
      // stratiform-cloudborne aerosol; else false
      const bool is_strat_cloudborne = false;
      if (0 == mam_prevap_resusp_optcc) {
        // wetdepa_v2 is almost a no-opt if 0==mam_prevap_resusp_optcc
        const int prevap_resusp_optcc = 0;
        wetdep::wetdepa_v2(
            dt, pdel[k], cmfdqr[k], evapc[k], dlf[k], conicw[k], prain[k],
            evapr[k], totcond[k], cldt[k], cldcu[k], cldvcu[k], cldvst[k],
            sol_factb[k], sol_facti[k], sol_factic[k], prevap_resusp_optcc,
            is_strat_cloudborne, scavcoef, f_act_conv[k], tracer, qqcw_tmp,
            precabs[k], precabc[k], scavabs[k], scavabc[k], precabs_base[k],
            precabc_base[k], precnums_base[k], precnumc_base[k], scavt[k],
            bcscavt[k], rcscavt[k], rsscavt);
      } else {
        wetdep::wetdepa_v2(
            dt, pdel[k], cmfdqr[k], evapc[k], dlf[k], conicw[k], prain[k],
            evapr[k], totcond[k], cldt[k], cldcu[k], cldvcu[k], cldvst[k],
            sol_factb[k], sol_facti[k], sol_factic[k], mam_prevap_resusp_optcc,
            is_strat_cloudborne, scavcoef, f_act_conv[k], tracer, qqcw_tmp,
            precabs[k], precabc[k], scavabs[k], scavabc[k], precabs_base[k],
            precabc_base[k], precnums_base[k], precnumc_base[k], scavt[k],
            bcscavt[k], rcscavt[k], rsscavt);
      }
      // resuspension goes to coarse mode
      const bool update_dqdt = true;
      aero_model::calc_resusp_to_coarse(mm, update_dqdt, rcscavt[k], rsscavt,
                                        scavt[k], rtscavt_sv_k);

    } else { // if (lphase == 2)
      // There is no cloud-borne aerosol water in the model, so this
      // code block should NEVER execute for lspec =
      // nspec_amode(m)+1 (i.e., jnummaswtr = 2). The code only
      // worked because the "do lspec" loop cycles when lspec =
      // nspec_amode(m)+1, but that does not make the code correct.
      // FIXME: Not sure if this is a bug or not as qqcw_tmp seem
      // different from the previous call and qqcw_tmp is always
      // zero. May need further check.  - Shuaiqi Tang in
      // refactoring for MAM4xx
      const Real tracer = qqcw(k, mm);
      const Real qqcw_tmp = 0.0;

      // resuspension, stratiform [kg/kg/s]
      Real rsscavt = 0.0;
      Real scavcoef = 0.0;
      if (jnv)
        scavcoef = (1 == jnv) ? scavcoefnum[k] : scavcoefvol[k];

      const bool is_strat_cloudborne = true;
      if (0 == mam_prevap_resusp_optcc) {
        // wetdepa_v2 is almost a no-opt if 0==mam_prevap_resusp_optcc
        const int prevap_resusp_optcc = 0;
        wetdep::wetdepa_v2(
            dt, pdel[k], cmfdqr[k], evapc[k], dlf[k], conicw[k], prain[k],
            evapr[k], totcond[k], cldt[k], cldcu[k], cldvcu[k], cldvst[k],
            sol_factb[k], sol_facti[k], sol_factic[k], prevap_resusp_optcc,
            is_strat_cloudborne, scavcoef, f_act_conv[k], tracer, qqcw_tmp,
            precabs[k], precabc[k], scavabs[k], scavabc[k], precabs_base[k],
            precabc_base[k], precnums_base[k], precnumc_base[k], scavt[k],
            bcscavt[k], rcscavt[k], rsscavt);
      } else {
        wetdep::wetdepa_v2(
            dt, pdel[k], cmfdqr[k], evapc[k], dlf[k], conicw[k], prain[k],
            evapr[k], totcond[k], cldt[k], cldcu[k], cldvcu[k], cldvst[k],
            sol_factb[k], sol_facti[k], sol_factic[k], mam_prevap_resusp_optcc,
            is_strat_cloudborne, scavcoef, f_act_conv[k], tracer, qqcw_tmp,
            precabs[k], precabc[k], scavabs[k], scavabc[k], precabs_base[k],
            precabc_base[k], precnums_base[k], precnumc_base[k], scavt[k],
            bcscavt[k], rcscavt[k], rsscavt);
      }
      // resuspension goes to coarse mode
      const bool update_dqdt = false;
      aero_model::calc_resusp_to_coarse(mm, update_dqdt, rcscavt[k], rsscavt,
                                        scavt[k], rtscavt_sv_k);
    }
  });
  team.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void update_q_tendencies(const ThreadTeam &team, const View2D &ptend_q,
                         const View1D &scavt, const int mm, const int nlev) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev),
                       [&](int k) { ptend_q(k, mm) += scavt[k]; });
}

// =============================================================================
KOKKOS_INLINE_FUNCTION
int get_aero_model_wetdep_work_len() {
  int work_len = 2 * mam4::nlev * pcnst +
                 // state_q, qqcw
                 36 * mam4::nlev +
                 // cldcu, cldst, evapc, cmfdqr, totcond, conicw,
                 // f_act_conv_coarse, f_act_conv_coarse_dust,
                 // f_act_conv_coarse_nacl, rain, cldv, cldvcu, cldvst,
                 // scavcoefnum, scavcoefvol
                 // sol_facti, sol_factic, sol_factb, f_act_conv,
                 // scavt, bcscavt, rcscavt,
                 // compute_q_tendencies_workspace(14, nlev)
                 2 * mam4::nlev * pcnst +
                 // ptend_q, rtscavt_sv
                 2 * pcnst +
                 // dqqcwdt
                 aero_model::pcnst * mam4::nlev;
  //  qsrflx_mzaer2cnvpr
  return work_len;
}
// =============================================================================
KOKKOS_INLINE_FUNCTION
void aero_model_wetdep(
    const ThreadTeam &team, const Atmosphere &atm, Prognostics &progs,
    Tendencies &tends, const Real dt, const Real scav_fraction_in_cloud_strat,
    const Real scav_fraction_in_cloud_conv,
    const Real scav_fraction_below_cloud_strat,
    const Real activation_fraction_in_cloud_conv,
    // inputs
    const ConstColumnView &cldt, const ConstColumnView &rprdsh,
    const ConstColumnView &rprddp, const ConstColumnView &evapcdp,
    const ConstColumnView &evapcsh, const ConstColumnView &dp_frac,
    const ConstColumnView &sh_frac, const ConstColumnView &icwmrdp,
    const ConstColumnView &icwmrsh, const ConstColumnView &evapr,
    const ConstColumnView &dlf, const ConstColumnView &prain,
    const View2D scavimptblnum, const View2D scavimptblvol,
    const CalcsizeData &calcsizedata,
    // in/out calcsize and water_uptake
    const View2D &wet_geometric_mean_diameter_i,
    const View2D &dry_geometric_mean_diameter_i, const View2D &qaerwat,
    const View2D &wetdens,
    // output
    const View1D &aerdepwetis, const View1D &aerdepwetcw, const View1D &work,
    const Int1D &isprx) {
  // cldn layer cloud fraction [fraction]; CLD

  // FIXME: do we need to set the variables inside of set_srf_wetdep ?
  // aerdepwetis aerosol_wet_deposition_interstitial;
  // aerdepwetcw aerosol_wet_deposition_cloud_water;

  // evapr evaporation_of_falling_precipitation;
  // shallow+deep convective detrainment [kg/kg/s]
  // dlf = diags.total_convective_detrainment;
  constexpr int ntot_amode = AeroConfig::num_modes();
  constexpr int nlev = mam4::nlev;
  constexpr int zero = 0.0;

  auto work_ptr = (Real *)work.data();
  // FIXME: is an input/output wet_geometric_mean_diameter_i ?
  // DGNUMWET
  // FIXME: is an input/output dry_geometric_mean_diameter_i ?
  // ColumnView dry_geometric_mean_diameter_i[ntot_amode];
  // // DGNUM
  // for (int m = 0; m < ntot_amode; ++m) {
  //   dry_geometric_mean_diameter_i[m] = ColumnView(work_ptr, mam4::nlev);
  //   work_ptr += mam4::nlev;
  // }

  // FIXME: is an input/output qaerwat ?
  // aerosol water [kg/kg]
  // qaerwat_idx    = pbuf_get_index('QAERWAT')
  // ColumnView qaerwat[ntot_amode];
  // for (int m = 0; m < ntot_amode; ++m) {
  //   qaerwat[m] = ColumnView(work_ptr, mam4::nlev);
  //   work_ptr += mam4::nlev;
  // }
  // // FIXME: is an input/output wetdens ?
  // // wet aerosol density [kg/m3]
  // // WETDENS_AP
  // ColumnView wetdens[ntot_amode];
  // for (int m = 0; m < ntot_amode; ++m) {
  //   wetdens[m] = ColumnView(work_ptr, mam4::nlev);
  //   work_ptr += mam4::nlev;
  // }

  View2D state_q(work_ptr, mam4::nlev, pcnst);
  work_ptr += mam4::nlev * pcnst;

  View2D qqcw(work_ptr, mam4::nlev, pcnst);
  work_ptr += mam4::nlev * pcnst;

  View1D cldcu(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D cldst(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D evapc(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D cmfdqr(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D totcond(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D conicw(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // inputs
  View1D f_act_conv_coarse(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D f_act_conv_coarse_dust(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D f_act_conv_coarse_nacl(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D rain(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // CHECK; is work array ?
  View1D cldv(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // CHECK; is work array ?
  View1D cldvcu(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // CHECK; is work array ?
  View1D cldvst(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D scavcoefnum(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mam4::nlev),
                       [&](int i) { scavcoefnum[i] = 0.0; });

  View1D scavcoefvol(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mam4::nlev),
                       [&](int i) { scavcoefvol[i] = 0.0; });

  View1D sol_facti(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D sol_factic(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D sol_factb(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D f_act_conv(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D scavt(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D bcscavt(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  View1D rcscavt(work_ptr, mam4::nlev);
  work_ptr += mam4::nlev;

  // FIXME: I need to get this variables from calcsize
  // need to connect ptend_q to tends
  View2D ptend_q(work_ptr, mam4::nlev, pcnst);
  work_ptr += mam4::nlev * pcnst;

  View2D rtscavt_sv(work_ptr, mam4::nlev, pcnst);
  work_ptr += pcnst * mam4::nlev;
  View2D dqqcwdt(work_ptr, mam4::nlev, aero_model::pcnst);
  work_ptr += aero_model::pcnst * mam4::nlev;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mam4::nlev), [&](int i) {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, pcnst), [&](int j) {
      rtscavt_sv(i, j) = zero;
      dqqcwdt(i, j) = zero;
    });
  });

  wetdep::zero_values(team, aerdepwetis, pcnst);
  wetdep::zero_values(team, aerdepwetcw, pcnst);

  View2D qsrflx_mzaer2cnvpr(work_ptr, aero_model::pcnst, 2);
  work_ptr += aero_model::pcnst * 2;

  const int num_tendencies_workspaces = 14;
  View2D compute_q_tendencies_workspace(work_ptr, num_tendencies_workspaces,
                                        mam4::nlev);
  work_ptr += num_tendencies_workspaces * mam4::nlev;

  /// error check
  const int workspace_used(work_ptr - work.data()),
      workspace_extent(work.extent(0));
  if (workspace_used > workspace_extent) {
    Kokkos::abort("Error aero_model_wetdep : workspace used is larger than it "
                  "is provided\n");
  }

  // inputs:
  // cldn; can we get it from atm?
  // cldt // layer cloud fraction [fraction] from pbuf_get_field
  // FIXME:
  constexpr int nwetdep = 1; // number of elements in wetdep_list

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mam4::nlev),
                       [&](int kk) { isprx[kk] = 0.0; });
  // inputs
  // Compute variables needed for convproc unified convective transport
  // rprdsh // pbuf_get_field rain production, shallow convection [kg/kg/s]
  // rprddp // pbuf_get_field rain production, deep convection [kg/kg/s]
  // evapcdp // pbuf_get_field Evaporation rate of shallow convective
  // precipitation >=0. [kg/kg/s] evapcsh // pbuf_get_field Evaporation rate of
  // deep    convective precipitation >=0. [kg/kg/s]

  // icwmrdp in cloud water mixing ratio, deep convection [kg/kg]
  // icwmrsh in cloud water mixing ratio, shallow convection [kg/kg]

  // inputs
  // dp_frac Deep convective cloud fraction [fraction]
  // sh_frac Shallow convective cloud fraction [fraction]
  // dp_ccf
  // sh_ccf

  // ouputs:
  // rprdshsum
  // rprddpsum
  // evapcdpsum
  // evapcshsum

  //
  ConstColumnView temperature = atm.temperature;
  ConstColumnView pmid = atm.pressure;
  ConstColumnView pdel = atm.hydrostatic_dp; // layer thickness (Pa)
  ConstColumnView q_liq = atm.liquid_mixing_ratio;
  ConstColumnView q_ice = atm.ice_mixing_ratio;

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev), [&](int kk) {
    // copy data from prog to stateq
    const auto state_q_kk = ekat::subview(state_q, kk);
    const auto qqcw_kk = ekat::subview(qqcw, kk);
    const auto ptend_q_kk = ekat::subview(ptend_q, kk);
    utils::extract_stateq_from_prognostics(progs, atm, state_q_kk, kk);
    utils::extract_qqcw_from_prognostics(progs, qqcw_kk, kk);
    utils::extract_ptend_from_tendencies(tends, ptend_q_kk, kk);
  });
  team.team_barrier();

  //
  // Do calculations of mode radius and water uptake if:
  // 1) modal aerosols are affecting the climate, or
  // 2) prognostic modal aerosols are enabled
  // If not using prognostic aerosol call the diagnostic version

  // Calculate aerosol size distribution parameters
  // for prognostic modal aerosols the transfer of mass between aitken and
  // accumulation modes is done in conjunction with the dry radius calculation
  // compute calcsize and

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, nlev), [&](int kk) {
    const auto state_q_kk = ekat::subview(state_q, kk);
    const auto qqcw_kk = ekat::subview(qqcw, kk);
    auto ptend_q_kk = ekat::subview(ptend_q, kk);
    Real dgnumwet_m_kk[ntot_amode] = {};
    // wetdens and qaerwat are input/ouput to water_uptake
    Real qaerwat_m_kk[ntot_amode] = {};
    Real wetdens_kk[ntot_amode] = {};
    // dgncur_a is aerosol particle diameter and is an input to
    // calcsize. But calcsize reset its value.
    Real dgnumdry_m_kk[ntot_amode] = {};
    for (int imode = 0; imode < ntot_amode; imode++) {
      dgnumwet_m_kk[imode] = wet_geometric_mean_diameter_i(imode, kk);
      dgnumdry_m_kk[imode] = dry_geometric_mean_diameter_i(imode, kk);
      qaerwat_m_kk[imode] = qaerwat(imode, kk);
      wetdens_kk[imode] = wetdens(imode, kk);
    }

    {
      Real dgncur_c_kk[ntot_amode] = {};
      // Real dqqcwdt_kk[pcnst] = {};
      auto dqqcwdt_kk = ekat::subview(dqqcwdt, kk);
      //  Calculate aerosol size distribution parameters and aerosol water
      //  uptake for prognostic aerosols
      modal_aero_calcsize::modal_aero_calcsize_sub(
          // Inputs
          state_q_kk, qqcw_kk, dt, calcsizedata,
          // Outputs
          dgnumdry_m_kk, dgncur_c_kk, ptend_q_kk, dqqcwdt_kk);
      // NOTE: dgnumdry_m_kk is interstitial dry diameter size and
      // dgncur_c_kk is cloud borne dry diameter size

      // update could aerosol.
      if (calcsizedata.update_mmr) {
        // Note: it only needs to update aerosol variables.
        for (int i = utils::aero_start_ind(); i < pcnst; ++i) {
          qqcw(kk, i) = mam4::max(zero, qqcw(kk, i) + dqqcwdt_kk[i] * dt);
        }
      } // end update could aerosols.

      mam4::water_uptake::modal_aero_water_uptake_dr(
          // inputs
          state_q_kk.data(), temperature(kk), pmid(kk), cldt(kk), dgnumdry_m_kk,
          // outputs
          dgnumwet_m_kk, qaerwat_m_kk, wetdens_kk);
    }

    // team.team_barrier();

    // save diameters, we use them in wet_dep.
    for (int imode = 0; imode < ntot_amode; imode++) {
      wet_geometric_mean_diameter_i(imode, kk) = dgnumwet_m_kk[imode];
      dry_geometric_mean_diameter_i(imode, kk) = dgnumdry_m_kk[imode];
      qaerwat(imode, kk) = qaerwat_m_kk[imode];
      wetdens(imode, kk) = wetdens_kk[imode];
    }
  }); // klev parallel_for loop

  team.team_barrier();

  // skip wet deposition if nwetdep is non-positive
  if (nwetdep < 1)
    return;
  {

    // // change mode order as mmode_loop_aa loops in a different order
    const int mode_order_change[4] = {0, 1, 3, 2};

    const int jaerowater = 2;

    // cumulus cloud fraction =  dp_frac + sh_frac
    wetdep::sum_values(team,                    // input
                       cldcu,                   // output
                       dp_frac, sh_frac, nlev); // inputs

    // total cloud fraction [fraction] = dp_ccf + sh_ccf
    // Stratiform cloud fraction cldst  = cldt - cldcu  Stratiform cloud
    // fraction
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlev),
                         [&](int k) { cldst[k] = cldt[k] - cldcu[k]; });

    // FIXME: where does eq come from?
    // FIXME: in fortran code cldt is equal to cln
    // wetdep::sum_values(team, cldt, dp_ccf, sh_ccf, nlev);
    // evaporation from convection (deep + shallow)
    wetdep::sum_values(team,                    // input
                       evapc,                   // output
                       evapcsh, evapcdp, nlev); // inputs

    // dq/dt due to convective cmfdqr =  rprddp + rprdsh
    wetdep::sum_values(team,                  // input
                       cmfdqr,                // output
                       rprddp, rprdsh, nlev); // inputs
    // rate of conversion of condensate to precipitation [kg/kg/s].
    // wetdep::sum_values(team, prain, icwmrdp, icwmrsh, nlev);
    // wetdep::sum_values(team, prain, icwmrdp, icwmrsh, nlev);
    // total condensate (ice+liq) [kg/kg] = q_liq + q_ice
    wetdep::sum_values(team,                // input
                       totcond,             // output
                       q_liq, q_ice, nlev); // input
    // sum deep and shallow convection contributions
    wetdep::sum_deep_and_shallow(team,                               // input
                                 conicw,                             // output
                                 icwmrdp, dp_frac, icwmrsh, sh_frac, // inputs
                                 nlev);                              // inputs
    team.team_barrier(); // for cldst
    // Estimate the cloudy volume which is occupied by rain or cloud water
    wetdep::cloud_diagnostics(team, temperature, pmid, pdel, cmfdqr, evapc,
                              cldt, cldcu, cldst, evapr, prain,
                              // outputs
                              cldv, cldvcu, cldvst, rain,
                              // inputs
                              nlev);

    team.team_barrier(); // for cldcu

    // calculate the mass-weighted sol_factic for coarse mode species
    // set the mass-weighted sol_factic for coarse mode species.
    wetdep::set_f_act(
        // input
        team,
        // outputs
        isprx.data(), f_act_conv_coarse, f_act_conv_coarse_dust,
        f_act_conv_coarse_nacl,
        // inputs
        pdel, prain, cmfdqr, evapr, state_q, ptend_q, dt, nlev);
    // main loop over aerosol modes
    for (int mtmp = 0; mtmp < AeroConfig::num_modes(); ++mtmp) {
      // for mam4, do accum, aitken, pcarbon, then coarse
      // so change the order of 2 and 3 here
      // for mam4:
      // do   accum = 0,
      // then aitken = 1,
      // then pcarbon - 3,
      // then coarse = 2
      const int imode = mode_order_change[mtmp];

      // loop over interstitial (1) and cloud-borne (2) forms
      // BSINGH (09/12/2014):Do cloudborne first for unified convection
      // scheme so that the resuspension of cloudborne can be saved then
      // applied to interstitial (RCE)

      // do cloudborne (2) first then interstitial (1)
      for (int lphase = 2; 1 <= lphase; --lphase) {
        team.team_barrier();
        if (lphase == 1) { // interstial aerosol
          // Computes lookup table for aerosol impaction/interception scavenging
          // rates
          wetdep::modal_aero_bcscavcoef_get(
              // inputs
              team, wet_geometric_mean_diameter_i, isprx.data(), scavimptblvol,
              scavimptblnum,
              // outputs
              scavcoefnum, scavcoefvol,
              // inputs
              imode, nlev);
          team.team_barrier();
        }
        // define sol_factb and sol_facti values, and f_act_conv
        wetdep::define_act_frac(
            // input
            team,
            // outputs
            sol_facti, sol_factic, sol_factb, f_act_conv,
            // inputs
            lphase, imode, nlev, scav_fraction_in_cloud_strat,
            scav_fraction_in_cloud_conv, scav_fraction_below_cloud_strat,
            activation_fraction_in_cloud_conv);

        // REASTER 08/12/2015 - changed ordering (mass then number) for
        // prevap resuspend to coarse loop over number + chem constituents +
        // water index for aerosol number / chem-mass / water-mass

        team.team_barrier();
        for (int lspec = 0; lspec < num_species_mode(imode) + 2; ++lspec) {
          int mm = 0, jnv = 0, jnummaswtr = 0;
          aero_model::index_ordering(lspec, imode, lphase, mm, jnv, jnummaswtr);
          // bypass wet aerosols
          if (0 <= mm && jnummaswtr != jaerowater) {
            // The following call mimics wetdepa_v2 subroutine call in
            // aero_model.F90
            //
            team.team_barrier();
            wetdep::compute_q_tendencies( // tendencies are in scavt
                team, nlev, f_act_conv, f_act_conv_coarse,
                f_act_conv_coarse_dust, f_act_conv_coarse_nacl, scavcoefnum,
                scavcoefvol, totcond, cmfdqr, conicw, evapc, evapr, prain, dlf,
                cldt, cldcu, cldst, cldvst, cldvcu, sol_facti, sol_factic,
                sol_factb,
                // outputs
                scavt, bcscavt, rcscavt, rtscavt_sv,
                // inputs
                state_q, qqcw,
                // outputs
                ptend_q,
                // inputs
                pdel, dt, jnummaswtr, jnv, mm, lphase, imode, lspec,
                // scratchspace
                compute_q_tendencies_workspace);
            team.team_barrier();

            // Note: update tendencies only in lphase == 1
            if (lphase == 1) {
              // Update ptend_q from the tendency, scavt
              wetdep::update_q_tendencies(team,             // input
                                          ptend_q,          // input
                                          scavt, mm, nlev); // inputs
            }
            if (lphase == 1) {
              aerdepwetis[mm] = aero_model::calc_sfc_flux(team,        // input
                                                          scavt,       // input
                                                          pdel, nlev); // inputs
            } else // if (lphase == 2)
            {
              aerdepwetcw[mm] = aero_model::calc_sfc_flux(team,        // input
                                                          scavt,       // input
                                                          pdel, nlev); // inputs
              team.team_barrier();
              Kokkos::parallel_for(
                  Kokkos::TeamVectorRange(team, nlev),
                  [&](int kk) { qqcw(kk, mm) += scavt(kk) * dt; });
            }
            team.team_barrier();
#if 0
            // Note: Commenting it out because it produces unused variable warnings.
            Real rprdshsum = aero_model::calc_sfc_flux(team, rprdsh, pdel, nlev);
            Real rprddpsum = aero_model::calc_sfc_flux(team, rprddp, pdel, nlev);
            Real evapcdpsum = aero_model::calc_sfc_flux(team, evapcdp, pdel, nlev);
            Real evapcshsum = aero_model::calc_sfc_flux(team, evapcsh, pdel, nlev);

            // NOTE. Adding this team_barrier fixed one race condition.
            team.team_barrier();
            const Real sflxbc =
                aero_model::calc_sfc_flux(team, bcscavt, pdel, nlev);
            const Real sflxec =
                aero_model::calc_sfc_flux(team, rcscavt, pdel, nlev);

            // apportion convective surface fluxes to deep and shallow
            // conv this could be done more accurately in subr wetdepa
            // since deep and shallow rarely occur simultaneously, and
            // these fields are just diagnostics, this approximate method
            // is adequate only do this for interstitial aerosol, because
            // conv clouds to not affect the stratiform-cloudborne
            // aerosol.
            // NOTE. Adding this team_barrier fixed one race condition.
            team.team_barrier();

            // FIXME: The following code is causing race condition errors in the
            // computer-sanitizer.
            //  I commented it out because we do not need it in the emaxx-mam4xx
            //  interface.
            {
              Real sflxbcdp, sflxecdp;
              aero_model::apportion_sfc_flux_deep(rprddpsum, rprdshsum,
                                                evapcdpsum, evapcshsum, sflxbc,
                                                sflxec, sflxbcdp, sflxecdp);

              // when ma_convproc_intr is used, convective in-cloud wet
              // removal is done there the convective (total and deep)
              // precip-evap-resuspension includes in- and below-cloud
              // contributions, so pass the below-cloud contribution to
              // ma_convproc_intr
              //
              // NOTE: ma_convproc_intr no longer uses these
              qsrflx_mzaer2cnvpr(mm, 0) = sflxec;
              qsrflx_mzaer2cnvpr(mm, 1) = sflxecdp;
            }
#endif
          }
        }
      }
    }
  }
  // make sure that ptend is updated in tendencies
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, nlev), [&](int kk) {
    const auto ptend_q_kk = ekat::subview(ptend_q, kk);
    const auto state_q_kk = ekat::subview(state_q, kk);
    const auto qqcw_kk = ekat::subview(qqcw, kk);
    utils::inject_qqcw_to_prognostics(qqcw_kk.data(), progs, kk);
    utils::inject_stateq_to_prognostics(state_q_kk.data(), progs, kk);
    utils::inject_ptend_to_tendencies(ptend_q_kk.data(), tends, kk);
  });
  team.team_barrier();

} // aero_model_wetdep

} // namespace wetdep

} // namespace mam4

#endif
