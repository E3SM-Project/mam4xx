// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_NUCLEATE_ICE_HPP
#define MAM4XX_NUCLEATE_ICE_HPP

#include "aero_config.hpp"
#include "aero_modes.hpp"
#include "atmosphere.hpp"
#include "conversions.hpp"
#include "mam4_math.hpp"
#include "mam4_types.hpp"
#include "surface.hpp"
#include "wv_sat_methods.hpp"

#include <ekat_math_utils.hpp>

namespace mam4 {

namespace nucleate_ice {

/*-------------------------------------------------------------------------------
Purpose:
 A parameterization of ice nucleation.

 *** This module is intended to be a "portable" code layer.  Ideally it should
 *** not contain any use association of modules that belong to the model
framework.


Method:
 The current method is based on Liu & Penner (2005) & Liu et al. (2007)
 It related the ice nucleation with the aerosol number, temperature and the
 updraft velocity. It includes homogeneous freezing of sulfate & immersion
 freezing on mineral dust (soot disabled) in cirrus clouds, and
 Meyers et al. (1992) deposition nucleation in mixed-phase clouds

 The effect of preexisting ice crystals on ice nucleation in cirrus clouds is
included,  and also consider the sub-grid variability of temperature in cirrus
clouds,  following X. Shi et al. ACP (2014).

 Ice nucleation in mixed-phase clouds now uses classical nucleation theory
(CNT),  follows Y. Wang et al. ACP (2014), Hoose et al. (2010).

Authors:
 Xiaohong Liu, 01/2005, modifications by A. Gettelman 2009-2010
 Xiangjun Shi & Xiaohong Liu, 01/2014.

 With help from C. C. Chen and B. Eaton (2014)
-------------------------------------------------------------------------------*/

KOKKOS_INLINE_FUNCTION
void calculate_regm_nucleati(const Real w_vlc, const Real Na, Real &regm) {
  /*-------------------------------------------------------------------------------
  Calculate temperature regime for ice nucleation based on
  Eq. 4.5 in Liu & Penner (2005), Meteorol. Z.
  -------------------------------------------------------------------------------*/

  // w_vlc            (updraft) vertical velocity [m/s]
  // Na               aerosol number concentration [#/cm^3]
  // regm             threshold temperature [C]

  const Real lnNa = mam4::log(Na);
  // BAD CONSTANT
  const Real A_coef = -Real(1.4938) * lnNa + Real(12.884);
  const Real B_coef = -Real(10.41) * lnNa - Real(67.69);

  regm = A_coef * mam4::log(w_vlc) + B_coef;
} // end calculate_regm_nucleati

KOKKOS_INLINE_FUNCTION
void calculate_RHw_hf(const Real temperature, const Real lnw, Real &RHw) {
  /*-------------------------------------------------------------------------------
  Calculate threshold relative humidity with respective to water (RHw) based
  on Eq. 3.1 in Liu & Penner (2005), Meteorol. Z.
  -------------------------------------------------------------------------------*/

  // temperature     temperature [C]
  // lnw             ln of vertical velocity
  // RHw             relative humidity threshold
  // NOTE(mjs): This math looks very odd to me

  const Real A_coef = Real(6.0e-4) * lnw + Real(6.6e-3); // this is ~7e-3
  const Real B_coef = Real(6.0e-2) * lnw + Real(1.052);  // this is ~1
  const Real C_coef = Real(1.68) * lnw + Real(129.35);   // this is ~131

  RHw = (A_coef * temperature * temperature + B_coef * temperature + C_coef) *
        Real(0.01);
  // Thus,
  // RHw ~= (7e-3 * temperature^2 + 1 * temperature + 131) * 0.01
  // NOTE: assuming the above, for temperature in [-160, -40],
  // RHw is in [0.9529, 1.5020]
} // end calculate_RHw_hf

KOKKOS_INLINE_FUNCTION
void calculate_Ni_hf(const Real A1, const Real B1, const Real C1, const Real A2,
                     const Real B2, const Real C2, const Real temperature,
                     const Real lnw, const Real Na, Real &Ni) {
  /*-------------------------------------------------------------------------------
  Calculate number of ice crystals (Ni) based on
  Eq. 3.3 in Liu & Penner (2005), Meteorol. Z.
  -------------------------------------------------------------------------------*/

  // A1, B1, C1     Coefficients
  // A2, B2, C2     Coefficients
  // temperature    temperature [C]
  // lnw            ln of vertical velocity
  // Na             aerosol number concentrations [#/cm^3]
  // Ni             ice number concentrations [#/cm^3]

  const Real k1 = mam4::exp(A2 + B2 * temperature + C2 * lnw);
  const Real k2 = A1 + B1 * temperature + C1 * lnw;

  Ni = mam4::min(k1 * mam4::pow(Na, k2), Na);
} // end calculate_Ni_hf

KOKKOS_INLINE_FUNCTION
void hf(const Real temperature, const Real w_vlc, const Real RH, const Real Na,
        const Real subgrid, Real &Ni) {

  /*-------------------------------------------------------------------------------
  Calculate number of ice crystals by homogeneous freezing (Ni) based on
  Liu & Penner (2005), Meteorol. Z.
  -------------------------------------------------------------------------------*/

  // temperature     temperature [C]
  // w_vlc           (updraft) vertical velocity [m/s]
  // RH              unitless relative humidity
  // Na              aerosol number concentrations [#/cm^3]
  // Ni              ice number concentrations [#/cm^3]

  /*---------------------------------------------------------------------
  parameters
  ---------------------------------------------------------------------*/

  const Real A1_fast = 0.0231;
  const Real A21_fast = -1.6387; // (T > -64 deg)
  const Real A22_fast = -6.045;  // (T <= -64 deg)
  const Real B1_fast = -0.008;
  const Real B21_fast = -0.042; // (T > -64 deg)
  const Real B22_fast = -0.112; // (T <= -64 deg)
  const Real C1_fast = 0.0739;
  const Real C2_fast = 1.2372;

  const Real A1_slow = -0.3949;
  const Real A2_slow = 1.282;
  const Real B1_slow = -0.0156;
  const Real B2_slow = 0.0111;
  const Real B3_slow = 0.0217;
  const Real C1_slow = 0.120;
  const Real C2_slow = 2.312;

  /*---------------------------------------------------------------------
  local variables
  ---------------------------------------------------------------------*/
  const Real zero = 0;
  Real A2_fast, B2_fast, B4_slow = zero;
  Real lnw, RHw = zero;

  lnw = mam4::log(w_vlc);

  Ni = zero;

  calculate_RHw_hf(temperature, lnw, RHw);

  if ((temperature <= -Real(37.0)) && (RH * subgrid >= RHw)) {
    const Real regm = Real(6.07) * lnw - Real(55.0);

    if (temperature >= regm) {
      // fast-growth regime
      if (temperature > -Real(64.0)) //
      {
        A2_fast = A21_fast;
        B2_fast = B21_fast;
      } else {
        A2_fast = A22_fast;
        B2_fast = B22_fast;
      } // end temperature

      calculate_Ni_hf(A1_fast, B1_fast, C1_fast, A2_fast, B2_fast, C2_fast,
                      temperature, lnw, Na, Ni);

    } else {
      //  slow-growth regime
      B4_slow = B2_slow + B3_slow * lnw;

      calculate_Ni_hf(A1_slow, B1_slow, C1_slow, A2_slow, B4_slow, C2_slow,
                      temperature, lnw, Na, Ni);
    } // end temperature >= regm
  }   // end temperature <= -Real(37.0)

} // end hf

KOKKOS_INLINE_FUNCTION
void hetero(const Real temperature, const Real w_vlc, const Real Ns, Real &Nis,
            Real &Nid) {

  /*-------------------------------------------------------------------------------
  Calculate number of ice crystals by heterogeneous freezing (Nis) based on
  Eq. 4.7 in Liu & Penner (2005), Meteorol. Z.
  -----------------------------------------------------------------------------*/

  // temperature     temperature [C]
  // w_vlc           (updraft) vertical velocity [m/s]
  // Ns              aerosol concentrations [#/cm^3]
  // Nis             ice number concentrations [#/cm^3]
  // Nid             ice number concentrations [#/cm^3]

  /*---------------------------------------------------------------------
  parameters
  ---------------------------------------------------------------------*/

  const Real A11 = 0.0263;
  const Real A12 = -0.0185;
  const Real A21 = 2.758;
  const Real A22 = 1.3221;
  const Real B11 = -0.008;
  const Real B12 = -0.0468;
  const Real B21 = -0.2667;
  const Real B22 = -1.4588;

  const Real lnNs = mam4::log(Ns);
  const Real lnw = mam4::log(w_vlc);

  // ice from immersion nucleation (cm^-3)

  const Real B_coef = (A11 + B11 * lnNs) * lnw + (A12 + B12 * lnNs);
  const Real C_coef = A21 + B21 * lnNs;

  Nis = mam4::exp(A22) * mam4::pow(Ns, B22) * mam4::exp(B_coef * temperature) *
        mam4::pow(w_vlc, C_coef);
  Nis = mam4::min(Nis, Ns);
  // FIXME: Mention that this variables is set to zero in PR
  // don't include deposition nucleation for cirrus clouds when T < -37C
  Nid = Real(0.0);

} // hetero

} // end namespace nucleate_ice

/// @class nucleate_ice
/// This class implements MAM4's nucleate_ice parameterization.
class NucleateIce {
public:
  // nucleate_ice-specific configuration
  struct Config {
    // In Fortran code _nucleate_ice_subgrid is read from a file.
    Real _nucleate_ice_subgrid;
    // ice nucleation SO2 size threshold for aitken mode
    Real _so4_sz_thresh_icenuc;
    Config(const Real nucleate_ice_subgrid = 120,
           const Real so4_sz_thresh_icenuc = 8.0e-8)
        : _nucleate_ice_subgrid(nucleate_ice_subgrid),
          _so4_sz_thresh_icenuc(so4_sz_thresh_icenuc) {}
    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;
  };

private:
  Config config_;
  Real _alnsg_amode_aitken, _num_m3_to_cm3, _so4_sz_thresh_icenuc, _mincld,
      _nucleate_ice_subgrid;

public:
  AeroSpeciesView aero_species;

  // name--unique name of the process implemented by this class
  const char *name() const { return "MAM4 nucleate_ice"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &nucleate_ice_config = Config()) {

    using mam4::log;

    aero_species = aero_config.aero_species;

    _nucleate_ice_subgrid = nucleate_ice_config._nucleate_ice_subgrid;

    _num_m3_to_cm3 = 1.0e-6;
    // BAD CONSTANT
    // FIXME
    // std::numeric_limits<Real>::max()
    // this values is from a txt file
    _so4_sz_thresh_icenuc = nucleate_ice_config._so4_sz_thresh_icenuc;
    // huge(1.0_r8)
    // minimum allowed cloud fraction
    // BAD CONSTANT
    _mincld = 0.0001;

    const int aitken_idx = int(ModeIndex::Aitken);
    _alnsg_amode_aitken = mam4::log(modes(aitken_idx).mean_std_dev);

  } // end(init)

  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Surface &surface,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const {

    using mam4::erf;
    using mam4::log;
    using mam4::sqrt;

    const int nk = atmosphere.num_levels();
    const Real tmelt_m_five = Constants::freezing_pt_h2o - 5;
    const int coarse_idx = int(ModeIndex::Coarse);
    const int aitken_idx = int(ModeIndex::Aitken);

    auto &coarse_dust = prognostics.q_aero_i[coarse_idx][aerosol_index_for_mode(
        ModeIndex::Coarse, AeroId::DST)];
    auto &coarse_nacl = prognostics.q_aero_i[coarse_idx][aerosol_index_for_mode(
        ModeIndex::Coarse, AeroId::NaCl)];
    auto &coarse_so4 = prognostics.q_aero_i[coarse_idx][aerosol_index_for_mode(
        ModeIndex::Coarse, AeroId::SO4)];

    auto &coarse_mom = prognostics.q_aero_i[coarse_idx][aerosol_index_for_mode(
        ModeIndex::Coarse, AeroId::MOM)];
    auto &coarse_bc = prognostics.q_aero_i[coarse_idx][aerosol_index_for_mode(
        ModeIndex::Coarse, AeroId::BC)];
    auto &coarse_pom = prognostics.q_aero_i[coarse_idx][aerosol_index_for_mode(
        ModeIndex::Coarse, AeroId::POM)];
    auto &coarse_soa = prognostics.q_aero_i[coarse_idx][aerosol_index_for_mode(
        ModeIndex::Coarse, AeroId::SOA)];

    auto &num_coarse = prognostics.n_mode_i[coarse_idx];
    auto &num_aitken = prognostics.n_mode_i[aitken_idx];

    // mode dry radius [m]
    auto &dgnum_aitken = diagnostics.dry_geometric_mean_diameter_i[aitken_idx];
    // updraft velocity for ice nucleation [m/s]
    auto &wsubi = atmosphere.updraft_vel_ice_nucleation;
    // could fraction [unitless]
    auto &ast = atmosphere.cloud_fraction;
    const Real subgrid = _nucleate_ice_subgrid;

    // number of activated aerosol for ice nucleation [#/kg]
    // output number conc of ice nuclei due to heterogeneous freezing [1/m3]
    auto &nihf = diagnostics.icenuc_num_hetfrz;
    // output number conc of ice nuclei due to immersion freezing (hetero nuc)
    // [1/m3]
    auto &niimm = diagnostics.icenuc_num_immfrz;
    // output number conc of ice nuclei due to deposition nucleation (hetero
    // nuc) [1/m3]
    auto &nidep = diagnostics.icenuc_num_depnuc;
    // output number conc of ice nuclei due to meyers deposition [1/m3]
    auto &nimey = diagnostics.icenuc_num_meydep;

    // number of activated aerosol for ice nucleation (homogeneous freezing
    // only) [#/kg]
    auto &naai_hom = diagnostics.num_act_aerosol_ice_nucle_hom;
    // number of activated aerosol for ice nucleation [#/kg]
    auto &naai = diagnostics.num_act_aerosol_ice_nucle;

    const Real num_m3_to_cm3 = _num_m3_to_cm3;
    // ice nucleation SO2 size threshold for aitken mode
    // input value from e3sm is not huge.
    const Real so4_sz_thresh_icenuc = _so4_sz_thresh_icenuc;

    const Real mincld = _mincld;
    const Real alnsg_amode_aitken = _alnsg_amode_aitken;

    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nk), [&](int kk) {
      const Real temp = atmosphere.temperature(kk);
      if (temp < tmelt_m_five) {

        const Real zero = 0;
        const Real half = 0.5;
        const Real sqrt_two = sqrt(2.0);

        const Real pmid = atmosphere.pressure(kk);
        const Real air_density = conversions::density_of_ideal_gas(temp, pmid);

        // CHECK if this part of code is consistent with original code.
        // relative humidity [unitless]
        Real qv = atmosphere.vapor_mixing_ratio(kk);
        // very low temperature produces inf relhum
        Real es = zero;
        Real qs = zero;

        wv_sat_methods::wv_sat_qsat_water(temp, pmid, es, qs);
        const Real relhum = qv / qs;
        const Real icldm = mam4::max(ast(kk), mincld);

        // compute aerosol number for so4, soot, and dust with units #/cm^3
        // remove soot number, because it is set to zero
        Real so4_num = zero;
        Real dst3_num = zero;

        /* For modal aerosols, assume for the upper troposphere:
        soot = accumulation mode
        sulfate = aitken mode
        dust = coarse mode
        since modal has internal mixtures. */
        Real dmc = coarse_dust(kk) * air_density;
        Real ssmc = coarse_nacl(kk) * air_density;
        Real so4mc = coarse_so4(kk) * air_density;

        Real mommc = coarse_mom(kk) * air_density;
        Real bcmc = coarse_bc(kk) * air_density;
        Real pommc = coarse_pom(kk) * air_density;
        Real soamc = coarse_soa(kk) * air_density;

        if (dmc > zero) {
          const Real wght =
              dmc / (ssmc + dmc + so4mc + bcmc + pommc + soamc + mommc);
          dst3_num = wght * num_coarse(kk) * air_density * num_m3_to_cm3;
        } // end dmc

        if (dgnum_aitken(kk) > zero) {
          // only allow so4 with D > 0.1 um in ice nucleation
          so4_num =
              num_aitken(kk) * air_density * num_m3_to_cm3 *
              (half -
               half * erf(mam4::log(so4_sz_thresh_icenuc / dgnum_aitken(kk)) /
                          (sqrt_two * alnsg_amode_aitken)));
        } // end dgnum_aitken

        so4_num = mam4::max(zero, so4_num);

        // Real naai = zero;

        nucleati(wsubi(kk), temp, pmid, relhum, icldm, air_density, so4_num,
                 dst3_num, subgrid,
                 // outputs
                 naai(kk), nihf(kk), niimm(kk), nidep(kk), nimey(kk));

        // QUESTION why nihf instead of naai
        naai_hom(kk) = nihf(kk);
        // is naai not saved?

        // output activated ice (convert from #/kg -> #/m3)
        // QUESTION: note that these variables are divided by rho in
        // nucleati
        nihf(kk) *= air_density;
        niimm(kk) *= air_density;
        nidep(kk) *= air_density;
        nimey(kk) *= air_density;

      } // end temp
    }); // kokkos::parfor(k)
  }

public:
  KOKKOS_INLINE_FUNCTION
  void nucleati( // inputs
      const Real wbar, const Real tair, const Real pmid, const Real relhum,
      const Real cldn, const Real rhoair, const Real so4_num,
      const Real dst3_num,
      // inputs
      const Real
          subgrid, // Subgrid scale factor on relative humidity (dimensionless)
      // outputs
      Real &nuci, Real &onihf, Real &oniimm, Real &onidep, Real &onimey) const {

    using mam4::pow;
    /*---------------------------------------------------------------
    Purpose:
     The parameterization of ice nucleation.

    Method: The current method is based on Liu & Penner (2005)
     It related the ice nucleation with the aerosol number, temperature and
    the  updraft velocity. It includes homogeneous freezing of sulfate,
    immersion  freezing of soot, and Meyers et al. (1992) deposition
    nucleation

    Authors: Xiaohong Liu, 01/2005, modifications by A. Gettelman 2009-2010
    ---------------------------------------------------------------- */

    // Input Arguments
    // wbar        grid cell mean (updraft) vertical velocity [m/s]
    // tair        temperature [K]
    // pmid        pressure at layer midpoints [pa]
    // relhum      relative humidity with respective to liquid [unitless]
    // cldn        new value of cloud fraction    [fraction]
    // rhoair      air density [kg/m3]
    // so4_num     so4 aerosol number [#/cm^3]
    // dst3_num     dust aerosol number [#/cm^3]

    // Output Arguments
    // nuci       ice number nucleated [#/kg]
    // onihf      nucleated number from homogeneous freezing of so4 [#/kg]
    // oniimm     nucleated number from immersion freezing [#/kg]
    // onidep     nucleated number from deposition nucleation [#/kg]
    // onimey     nucleated number from deposition nucleation  (meyers: mixed
    // phase) [#/kg]

    // Local workspace
    Real zero = 0;
    Real nihf = zero;  //                     nucleated number from homogeneous
                       //                     freezing of so4 [#/cm^3]
    Real niimm = zero; //                     nucleated number from immersion
                       //                     freezing [#/cm^3]
    // NOTE: this gets set to zero in every logic branch below
    // and also within hetero()
    Real nidep = zero; //                     nucleated number from deposition
                       //                     nucleation [#/cm^3]
    // NOTE: this gets set to zero at the very end
    Real nimey = zero; //                    nucleated number from
    // deposition nucleation (meyers) [#/cm^3]
    Real n1 = zero;
    Real ni = zero; //                  nucleated number [#/cm^3]
    const Real tc =
        tair - Real(273.15); //                      air temperature [C]
    Real regm = zero;        //                    air temperature [C]

    // BAD CONSTANT
    const Real num_threshold = 1.0e-10;

    if (so4_num >= num_threshold && dst3_num >= num_threshold && cldn > zero) {
      if ((tc <= Real(-35.0)) && (relhum * wv_sat_methods::svp_water(tair) /
                                      wv_sat_methods::svp_ice(tair) * subgrid >=
                                  Real(1.2))) {
        // use higher RHi threshold
        nucleate_ice::calculate_regm_nucleati(wbar, dst3_num, regm);
        if (tc > regm) {
          // heterogeneous nucleation only
          // BAD CONSTANT
          if (tc < -Real(40) && wbar > Real(1.)) {
            // exclude T < -40 & W > 1 m/s from hetero.nucleation

            nucleate_ice::hf(tc, wbar, relhum, so4_num, subgrid, nihf);
            niimm = zero;
            nidep = zero;
            n1 = nihf;

          } else {

            nucleate_ice::hetero(tc, wbar, dst3_num, niimm, nidep);
            nihf = zero;
            n1 = niimm + nidep;

          } // end tc<Real(-40) ...
        } else if (tc < regm - Real(5.)) {
          // homogeneous nucleation only
          nucleate_ice::hf(tc, wbar, relhum, so4_num, subgrid, nihf);
          niimm = zero;
          nidep = zero;
          n1 = nihf;
        } else {
          // transition between homogeneous and heterogeneous: interpolate
          // in-between

          // BAD CONSTANT
          if (tc < -Real(40.) && wbar > Real(1.)) {
            // exclude T < -40 & W > 1 m/s from hetero. nucleation

            nucleate_ice::hf(tc, wbar, relhum, so4_num, subgrid, nihf);
            niimm = zero;
            nidep = zero;
            n1 = nihf;

          } else {

            nucleate_ice::hf(regm - Real(5.), wbar, relhum, so4_num, subgrid,
                             nihf);
            nucleate_ice::hetero(regm, wbar, dst3_num, niimm, nidep);

            if (nihf <= (niimm + nidep)) {
              n1 = niimm + nidep;
            } else {
              n1 = (niimm + nidep) *
                   mam4::pow((niimm + nidep) / nihf, (tc - regm) / Real(5.));

            } // end nihf <= (niimm + nidep)

          } // end tc < -40.

        } // end tc > regm

        ni = n1;

      } // end tc ...

    } // end so4_num ..

    /* deposition/condensation nucleation in mixed clouds (-37 < T < 0 C)
    (Meyers, 1992) this part is executed but is always replaced by 0, because
    CNT scheme takes over the calculation. use_hetfrz_classnuc is always true.
  */
    // FIXME OD: why adding zero to nuci? is something missing?
    // mjs: this whole thing is bizarre--add zero and if that makes it >= 1e4
    // or < 0, then make it zero? And this is the only thing that happens to
    // nuci in this process?
    nimey = zero;
    // BAD CONSTANT
    nuci = ni + nimey;
    if (nuci > Real(9999.) || nuci < zero) {
      nuci = zero;
    } // end

    const Real one_millon = 1.e+6;
    nuci = nuci * one_millon / rhoair; //  change unit from #/cm3 to #/kg
    onimey = nimey * one_millon / rhoair;
    onidep = nidep * one_millon / rhoair;
    oniimm = niimm * one_millon / rhoair;
    onihf = nihf * one_millon / rhoair;

  } // end nucleati
};  // end class nucleate_ice
} // end namespace mam4

#endif
