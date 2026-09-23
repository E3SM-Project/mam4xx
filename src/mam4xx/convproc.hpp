// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_CONVPROC_HPP
#define MAM4XX_CONVPROC_HPP

#include "aero_config.hpp"
#include "aero_model.hpp"
#include "aero_modes.hpp"
#include "atmosphere.hpp"
#include "mam4_math.hpp"
#include "ndrop.hpp"
#include "utils.hpp"
#include "wv_sat_methods.hpp"

namespace mam4 {

// Namespace for WetDep function until wetdep is implimented.
namespace WetDepTemp {
KOKKOS_INLINE_FUNCTION
Real faer_resusp_vs_fprec_evap_mpln(const Real fprec_evap, const int) {
  // --------------------------------------------------------------------------------
  // corresponding fraction of precipitation-borne aerosol flux that is
  // resuspended Options of assuming log-normal or marshall-palmer raindrop size
  // distribution note that these fractions are relative to the cloud-base
  // fluxes, and not to the layer immediately above fluxes
  // --------------------------------------------------------------------------------

  // faer_resusp_vs_fprec_evap_mpln ! [fraction]
  // in fprec_evap [fraction]
  // in jstrcnv - current only two options:
  //    1 for marshall-palmer distribution
  //    2 for log-normal distribution
  Real a01, a02, a03, a04, a05, a06, a07, a08, a09, x_lox_lin, y_lox_lin;
  // log-normal distribution
  a01 = 6.1944215103685640E-02;
  a02 = -2.0095166685965378E+00;
  a03 = 2.3882460251821236E+01;
  a04 = -1.2695611774753374E+02;
  a05 = 4.0086943562320101E+02;
  a06 = -7.4954272875943707E+02;
  a07 = 8.1701055892023624E+02;
  a08 = -4.7941894659538502E+02;
  a09 = 1.1710291076059025E+02;
  x_lox_lin = 1.0000000000000001E-01;
  y_lox_lin = 6.2227889828044350E-04;
  Real x_var, y_var;
  x_var = utils::min_max_bound(0.0, 1.0, fprec_evap);
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
} // namespace WetDepTemp
  //
namespace ndrop_od {

KOKKOS_INLINE_FUNCTION
void qsat(const Real t, const Real p, Real &es, Real &qs) {
  //  ------------------------------------------------------------------!
  // ! Purpose:                                                         !
  // !   Look up and return saturation vapor pressure from precomputed  !
  // !   table, then calculate and return saturation specific humidity. !
  // !   Optionally return various temperature derivatives or enthalpy  !
  // !   at saturation.                                                 !
  // !------------------------------------------------------------------!
  // Inputs
  // @param [in] t    ! Temperature
  // @param [in] p    ! Pressure
  // Outputs
  // @param [out] es  ! Saturation vapor pressure
  // @param [out] qs  ! Saturation specific humidity

  // Note. Fortran code uses a table lookup. In C++ version, we compute directly
  // from the function.
  es = wv_sat_methods::wv_sat_svp_trans(t);
  qs = wv_sat_methods::wv_sat_svp_to_qsat(es, p);
  // Ensures returned es is consistent with limiters on qs.
  es = mam4::min(es, p);
} // qsat

KOKKOS_INLINE_FUNCTION
void ndrop_int(Real exp45logsig[AeroConfig::num_modes()],
               Real alogsig[AeroConfig::num_modes()], Real &aten,
               Real num2vol_ratio_min_nmodes[AeroConfig::num_modes()],
               Real num2vol_ratio_max_nmodes[AeroConfig::num_modes()]) {
  const Real surften = 0.076;
  const Real t0 = 273; // reference temperature [K]
  const Real one = 1;
  const Real two = 2;
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

  // SHR_CONST_RHOFW   = 1.000e3_R8      ! density of fresh water     ~ kg/m^3
  const Real rhoh2o = Constants::density_h2o;
  const Real r_universal = Constants::r_gas * one_thousand;      //[J/K/kmole]
  const Real mwh2o = Constants::molec_weight_h2o * one_thousand; // [kg/kmol]
  // BAD CONSTANT
  aten = two * mwh2o * surften / (r_universal * t0 * rhoh2o);

} // end ndrop_int

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
                    Real fluxm[AeroConfig::num_modes()], Real &flux_fullact,
                    const Real smax_prescribed = mam4::max()) {
  // 	  !---------------------------------------------------------------------------------
  // !Calculates number, surface, and mass fraction of aerosols activated as CCN
  // !calculates flux of cloud droplets, surface area, and aerosol mass into
  // cloud !assumes an internal mixture within each of up to nmode multiple
  // aerosol modes !a gaussiam spectrum of updrafts can be treated.
  // !
  // !Units: SI (MKS)
  // !
  // !Reference: Abdul-Razzak and Ghan, A parameterization of aerosol
  // activation. !      2. Multiple aerosol types. J. Geophys. Res., 105,
  // 6837-6844.
  // !---------------------------------------------------------------------------------

  // input
  // @param [in] w_in      ! vertical velocity [m/s]
  // @param [in] wmaxf     ! maximum updraft velocity for integration [m/s]
  // @param [in] tair      ! air temperature [K]
  // @param [in] rhoair    ! air density [kg/m3]
  // @param [in] na(:)     ! aerosol number concentration [#/m3]
  // @param [in] nmode     ! number of aerosol modes
  // @param [in] volume(:) ! aerosol volume concentration [m3/m3]
  // @param [in] hygro(:)  ! hygroscopicity of aerosol mode [dimensionless]
  // @param [in] smax_prescribed  ! prescribed max. supersaturation for
  // secondary activation [fraction]

  // !output
  // @param [out] fn(:)        ! number fraction of aerosols activated
  // [fraction]
  // @param [out] fm(:)        ! mass fraction of aerosols activated [fraction]
  // @param [out] fluxn(:)     ! flux of activated aerosol number fraction into
  // cloud [m/s]
  // @param [out] fluxm(:)     ! flux of activated aerosol mass fraction into
  // cloud [m/s]
  // @param [out] flux_fullact ! flux of activated aerosol fraction assuming
  // 100% activation [m/s]

  // !---------------------------------------------------------------------------------
  // ! flux_fullact is used for consistency check -- this should match
  // (ekd(k)*zs(k)) ! also, fluxm/flux_fullact gives fraction of aerosol mass
  // flux ! that is activated
  // !---------------------------------------------------------------------------------
  const Real zero = 0;
  const Real one = 1;
  const Real sq2 = mam4::sqrt(2.);
  const Real two = 2;
  const Real three_fourths = 3. / 4.;
  const Real twothird = 2. / 3.;
  const Real half = 0.5;
  const Real small = 1.e-39;
  const Real t0 = 273; // reference temperature [K]

  constexpr int nmode = AeroConfig::num_modes();
  // BAD CONSTANT
  // return if aerosol number is negligible in the accumulation mode
  // FIXME use index of accumulation mode
  if (na[0] < 1.e-20) {
    return;
  }

  // return if vertical velocity is 0 or negative
  if (w_in <= zero) {
    return;
  }

  if (smax_prescribed <= 0.0)
    return;

  // BAD CONSTANT
  //  FIXME look for constant in ahero
  //  const Real SHR_CONST_MWDAIR  = 28.966;//       ! molecular weight dry air
  //  ~ kg/kmole const Real SHR_CONST_MWWV    = 18.016;//       ! molecular
  //  weight water vapor const Real  rair = SHR_CONST_RGAS/SHR_CONST_MWDAIR;//
  //  ! Dry air gas constant     ~ J/K/kg

  const Real rair = Constants::r_gas_dry_air;
  const Real rh2o = Constants::r_gas_h2o_vapor;
  const Real latvap =
      Constants::latent_heat_evap;          // latent heat of evaporation ~ J/kg
  const Real cpair = Constants::cp_dry_air; // specific heat of dry air ~ J/kg/K
  const Real gravit = Constants::gravity;   // acceleration of gravity ~ m/s^2
  // SHR_CONST_RHOFW   = 1.000e3_R8      ! density of fresh water     ~ kg/m^3
  const Real rhoh2o = Constants::density_h2o;
  const Real pi = Constants::pi;
  const Real p0 = 1013.25e2; //  ! reference pressure [Pa]

  const Real pres = rair * rhoair * tair; // pressure [Pa]
  // Obtain Saturation vapor pressure (es) and saturation specific humidity (qs)
  //  FIXME: check if we have implemented qsat
  //  water vapor saturation specific humidity [kg/kg]
  Real qs = zero;
  // ! saturation vapor pressure [Pa]
  Real es = zero;           //
  qsat(tair, pres, es, qs); // !es and qs are the outputs
  // ! change in qs with temperature  [(kg/kg)/T]
  const Real dqsdt = latvap / (rh2o * tair * tair) * qs;
  // [/m]
  const Real alpha =
      gravit * (latvap / (cpair * rh2o * tair * tair) - one / (rair * tair));
  // [m3/kg]
  const Real gamma = (one + latvap / cpair * dqsdt) / (rhoair * qs);
  // [s^(3/2)]
  // BAD CONSTANT
  const Real etafactor2max =
      1.e10 / mam4::pow((alpha * wmaxf),
                        1.5); // !this should make eta big if na is very small.
  // vapor diffusivity [m2/s]
  const Real diff0 = 0.211e-4 * (p0 / pres) * mam4::pow(tair / t0, 1.94);
  // ! thermal conductivity [J / (m-s-K)]
  const Real conduct0 =
      (5.69 + 0.017 * (tair - t0)) * 4.186e2 * 1.e-5; // !convert to J/m/s/deg
  // thermodynamic function [m2/s]
  const Real gthermfac =
      one /
      (rhoh2o / (diff0 * rhoair * qs) +
       latvap * rhoh2o / (conduct0 * tair) *
           (latvap / (rh2o * tair) - one)); // gthermfac is same for all modes
  const Real beta = two * pi * rhoh2o * gthermfac * gamma; //[m2/s]
  // nucleation w, but = w_in if wdiab == 0 [m/s]
  const Real wnuc = w_in;
  const Real alw = alpha * wnuc;                 // [/s]
  const Real etafactor1 = alw * mam4::sqrt(alw); // [/ s^(3/2)]
  // [unitless]
  const Real zeta = twothird * mam4::sqrt(alw) * aten / mam4::sqrt(gthermfac);

  Real amcube[nmode] = {}; // ! cube of dry mode radius [m3]

  Real etafactor2[nmode] = {};
  Real lnsm[nmode] = {};

  // critical supersaturation for number mode radius [fraction]
  Real smc[nmode] = {};

  Real eta[nmode] = {};
  // !Here compute smc, eta for all modes for maxsat calculation
  for (int imode = 0; imode < nmode; ++imode) {
    // BAD CONSTANT

    if (volume[imode] > small && na[imode] > small) {
      // !number mode radius (m)
      amcube[imode] =
          three_fourths * volume[imode] /
          (pi * exp45logsig[imode] * na[imode]); // ! only if variable size dist
      // !Growth coefficent Abdul-Razzak & Ghan 1998 eqn 16
      // !should depend on mean radius of mode to account for gas kinetic
      // effects !see Fountoukis and Nenes, JGR2005 and Meskhidze et al.,
      // JGR2006 !for approriate size to use for effective diffusivity.
      etafactor2[imode] = one / (na[imode] * beta * mam4::sqrt(gthermfac));
      // BAD CONSTANT
      if (hygro[imode] > 1.e-10) {
        smc[imode] =
            two * aten *
            mam4::sqrt(aten / (27. * hygro[imode] *
                               amcube[imode])); // ! only if variable size dist
      } else {
        // BAD CONSTANT
        smc[imode] = 100.;
      } // hygro
    } else {
      smc[imode] = one;
      etafactor2[imode] =
          etafactor2max; // ! this should make eta big if na is very small.
    }                    // volumne
    lnsm[imode] = mam4::log(smc[imode]); // ! only if variable size dist
    eta[imode] = etafactor1 * etafactor2[imode];
  } // end imode

  // Find maximum supersaturation
  // Use smax_prescribed if it is present; otherwise get smax from subr maxsat
  Real smax = smax_prescribed;
  if (smax_prescribed == mam4::max())
    ndrop::maxsat(zeta, eta, nmode, smc, smax);

  // FIXME [unitless] ? lnsmax maybe has units of log(unit of smax ([fraction]))
  const Real lnsmax = mam4::log(smax);

  // !Use maximum supersaturation to calculate aerosol activation output
  for (int imode = 0; imode < nmode; ++imode) {
    // ! [unitless]
    const Real arg_erf_n =
        twothird * (lnsm[imode] - lnsmax) / (sq2 * alogsig[imode]);

    fn[imode] = half * (one - mam4::erf(arg_erf_n)); //! activated number

    const Real arg_erf_m = arg_erf_n - 1.5 * sq2 * alogsig[imode];
    fm[imode] = half * (one - mam4::erf(arg_erf_m)); // !activated mass
    fluxn[imode] = fn[imode] * w_in; // !activated aerosol number flux
    fluxm[imode] = fm[imode] * w_in; // !activated aerosol mass flux
  }
  // FIXME: what is this??
  // is vertical velocity equal to flux of activated aerosol fraction assuming
  // 100% activation [m/s]?
  flux_fullact = w_in;

} // activate_modal

} // namespace ndrop_od

/// @class ConvProc
/// This class implements MAM4's ConvProc parameterization.
class ConvProc {
public:
  // maxd_aspectype = maximum allowable number of chemical species
  // in each aerosol mode.
  //
  static constexpr int maxd_aspectype = 14;
  // TODO: gas_pcnst number of "gas phase" species
  // This should come from the chemistry model being used and will
  // probably need to be dynamic.  This forces lmassptr_amode to
  // also be dynamic.
  // static constexpr int gas_pcnst = aero_model::pcnst;

  // ====================================================================================
  // The diagnostic arrays are twice the lengths of aero_model::pcnst
  // because cloudborne aerosols are appended after interstitial aerosols both
  // of which are of length gas_pcnst.
  static constexpr int pcnst_extd = 2 * aero_model::pcnst;

  // nucleation-specific configuration
  struct Config {

    // default constructor -- sets default values for parameters
    Config() {}

    Config(const Config &) = default;
    ~Config() = default;
    Config &operator=(const Config &) = default;

    bool convproc_do_aer = true;
    bool convproc_do_gas = false;
    int nlev = mam4::nlev;
    int ktop = 47; // BAD_CONSTANT: only true for nlev == 72
    int kbot = mam4::nlev - 1;
    // Flags are defined in "enum ConvProc::species_class".
    int species_class[aero_model::pcnst] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    int mmtoo_prevap_resusp[aero_model::pcnst] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 30, 32, 33, 31, 28, 29, 34, -3, 30, 33, 29, 34, -3,
        28, 29, 30, 31, 32, 33, 34, -3, 32, 31, 34, -3};
  };

  static constexpr int num_modes = AeroConfig::num_modes();
  static constexpr int num_aerosol_ids = AeroConfig::num_aerosol_ids();

  // identifies the type of
  // convection ("deep", "shcu")
  enum convtype { Deep, Uwsh };
  // Constantants for MAM species classes
  enum species_class {
    undefined = 0,  // spec_class_undefined
    cldphysics = 1, // spec_class_cldphysics
    aerosol = 2,    // spec_class_aerosol
    gas = 3,        // spec_class_gas
    other = 4       // spec_class_other
  };
  // Where lmapcc_val_num are defined in lmapcc_all
  //
  // numptr_amode(m) = gchm r-array index for the number mixing ratio
  // (particles/mole-air) for aerosol mode m that is in clear air or
  // interstitial are (but not in cloud water).  If zero or negative,
  // then number is not being simulated.
  KOKKOS_INLINE_FUNCTION
  static constexpr int numptr_amode(const int i) {
    const int numptr_amode[num_modes] = {22, 27, 35, 39};
    return numptr_amode[i];
  }
  // Use the same index for Q and QQCW arrays
  KOKKOS_INLINE_FUNCTION
  static constexpr int numptrcw_amode(const int i) { return numptr_amode(i); }

  // Where lmapcc_val_aer are defined in lmapcc_all
  //
  // lmassptr_amode(l,m) = gchm r-array index for the mixing ratio
  // (moles-x/mole-air) for chemical species l in aerosol mode m
  // that is in clear air or interstitial air (but not in cloud water).
  // If negative then number is not being simulated.
  KOKKOS_INLINE_FUNCTION
  static constexpr int lmassptr_amode(const int i, const int j) {
    const int lmassptr_amode[maxd_aspectype][num_modes] = {
        {15, 23, 28, 36}, {16, 24, 29, 37}, {17, 25, 30, 38}, {18, 26, 31, -1},
        {19, -1, 32, -1}, {20, -1, 33, -1}, {21, -1, 34, -1}, {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}};
    return lmassptr_amode[i][j];
  }
  // use the same index for Q and QQCW arrays
  //
  // lmassptrcw_amode(l,m) = gchm r-array index for the mixing ratio
  // (moles-x/mole-air) for chemical species l in aerosol mode m
  // that is currently bound/dissolved in cloud water
  KOKKOS_INLINE_FUNCTION
  static constexpr int lmassptrcw_amode(const int i, const int j) {
    return lmassptr_amode(i, j);
  }
  // lspectype_amode(l,m) = species type/i.d. for chemical species l
  // in aerosol mode m.  (0=sulfate, others to be defined)
  KOKKOS_INLINE_FUNCTION
  static constexpr int lspectype_amode(const int i, const int j) {
    const int lspectype_amode[maxd_aspectype][num_modes] = {
        {0, 0, 7, 3},     {3, 4, 6, 5},     {4, 6, 0, 8},     {5, 8, 5, -1},
        {7, -1, 3, -1},   {6, -1, 4, -1},   {8, -1, 8, -1},   {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1},
        {-1, -1, -1, -1}, {-1, -1, -1, -1}};
    return lspectype_amode[i][j];
  }

  // specdens_amode(l) = dry density (kg/m^3) of aerosol chemical species type l
  // This is indexed by the values returned from lspectype_amode and should
  // be simplified to just a direct call to aero_species(i).density
  // but there is the problem of the indexes being in a different order.
  //
  KOKKOS_INLINE_FUNCTION
  static Real specdens_amode(const AeroSpeciesView &aero_species, const int i) {
    // clang-format off
    const Real specdens_amode[maxd_aspectype] = {
      aero_species(int(AeroId::SO4)).density,
      mam4::max(),
      mam4::max(),
      aero_species(int(AeroId::POM)).density,
      aero_species(int(AeroId::SOA)).density,
      aero_species(int(AeroId::BC)).density,
      aero_species(int(AeroId::NaCl)).density,
      aero_species(int(AeroId::DST)).density,
      aero_species(int(AeroId::MOM)).density,
      0, 0, 0, 0, 0};
    // clang-format on
    return specdens_amode[i];
  }

  // specdens_amode(l) = dry density (kg/m^3) of aerosol chemical species type l
  // The same concerns specified for specdens_amode apply to spechygro.
  KOKKOS_INLINE_FUNCTION
  static Real spechygro(const AeroSpeciesView &aero_species, const int i) {
    // clang-format off
    const Real spechygro[maxd_aspectype] = {
       aero_species(int(AeroId::SO4)).hygroscopicity,
       mam4::max(),
       mam4::max(),
       aero_species(int(AeroId::POM)).hygroscopicity,
       0.1400000000e+00, // FIXME: should be aero_species(int(AeroId::SOA)).hygroscopicity
       aero_species(int(AeroId::BC)).hygroscopicity,
       aero_species(int(AeroId::NaCl)).hygroscopicity,
       0.6800000000e-01, // FIXME: should be aero_species(int(AeroId::DST)).hygroscopicity,
       aero_species(int(AeroId::MOM)).hygroscopicity,
       0, 0, 0, 0, 0};
    // clang-format on
    return spechygro[i];
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Real voltonumbhi_amode(const int i) {
    // BAD_CONSTANT - Will eventually be defined and retrieved from ndrop
    const Real voltonumbhi_amode[num_modes] = {
        0.4736279937e+19, 0.5026599108e+22, 0.6303988596e+16, 0.7067799781e+21};
    return voltonumbhi_amode[i];
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr Real voltonumblo_amode(const int i) {
    // BAD_CONSTANT - Will eventually be defined and retrieved from ndrop
    const Real voltonumblo_amode[num_modes] = {
        0.2634717443e+22, 0.1073313330e+25, 0.4034552701e+18, 0.7067800158e+24};
    return voltonumblo_amode[i];
  }

  enum Col1DViewInd {
    q = 0,
    mu,
    md,
    eudp,
    dudp,
    eddp,
    dddp,
    rhoair,
    zmagl,
    gath,
    chat,
    conu,
    cond,
    dconudt_wetdep,
    dconudt_activa,
    fa_u,
    dcondt,
    dcondt_wetdep,
    dcondt_prevap,
    dcondt_prevap_hist,
    dcondt_resusp,
    wd_flux,
    sumactiva,
    sumaqchem,
    sumprevap,
    sumprevap_hist,
    sumresusp,
    sumwetdep,
    dqdt,
    qnew,
    dlfdp,
    NumScratch
  };

private:
  Config config_;

  Kokkos::View<Real *> scratch1Dviews[NumScratch];

public:
  // name -- unique name of the process implemented by this class
  const char *name() const { return "MAM4 convproc"; }

  // init -- initializes the implementation with MAM4's configuration and with
  // a process-specific configuration.
  void init(const AeroConfig &aero_config,
            const Config &convproc_config = Config()) {
    // Set nucleation-specific config parameters.
    config_ = convproc_config;
    Kokkos::resize(scratch1Dviews[q], config_.nlev * aero_model::pcnst);
    Kokkos::resize(scratch1Dviews[mu], config_.nlev + 1);
    Kokkos::resize(scratch1Dviews[md], config_.nlev + 1);
    Kokkos::resize(scratch1Dviews[eudp], config_.nlev);
    Kokkos::resize(scratch1Dviews[dudp], config_.nlev);
    Kokkos::resize(scratch1Dviews[eddp], config_.nlev);
    Kokkos::resize(scratch1Dviews[dddp], config_.nlev);
    Kokkos::resize(scratch1Dviews[rhoair], config_.nlev);
    Kokkos::resize(scratch1Dviews[zmagl], config_.nlev);
    Kokkos::resize(scratch1Dviews[zmagl], config_.nlev);
    Kokkos::resize(scratch1Dviews[gath], (1 + config_.nlev) * pcnst_extd);
    Kokkos::resize(scratch1Dviews[chat], (1 + config_.nlev) * pcnst_extd);
    Kokkos::resize(scratch1Dviews[conu], (1 + config_.nlev) * pcnst_extd);
    Kokkos::resize(scratch1Dviews[cond], (1 + config_.nlev) * pcnst_extd);
    Kokkos::resize(scratch1Dviews[dconudt_wetdep],
                   (1 + config_.nlev) * pcnst_extd);
    Kokkos::resize(scratch1Dviews[dconudt_activa],
                   (1 + config_.nlev) * pcnst_extd);
    Kokkos::resize(scratch1Dviews[fa_u], config_.nlev);
    Kokkos::resize(scratch1Dviews[dcondt], config_.nlev * pcnst_extd);
    Kokkos::resize(scratch1Dviews[dcondt_wetdep], config_.nlev * pcnst_extd);
    Kokkos::resize(scratch1Dviews[dcondt_prevap], config_.nlev * pcnst_extd);
    Kokkos::resize(scratch1Dviews[dcondt_prevap_hist],
                   config_.nlev * pcnst_extd);
    Kokkos::resize(scratch1Dviews[dcondt_resusp], config_.nlev * pcnst_extd);
    Kokkos::resize(scratch1Dviews[wd_flux], pcnst_extd);
    Kokkos::resize(scratch1Dviews[sumactiva], pcnst_extd);
    Kokkos::resize(scratch1Dviews[sumaqchem], pcnst_extd);
    Kokkos::resize(scratch1Dviews[sumprevap], pcnst_extd);
    Kokkos::resize(scratch1Dviews[sumprevap_hist], pcnst_extd);
    Kokkos::resize(scratch1Dviews[sumresusp], pcnst_extd);
    Kokkos::resize(scratch1Dviews[sumwetdep], pcnst_extd);
    Kokkos::resize(scratch1Dviews[dqdt], config_.nlev * aero_model::pcnst);
    Kokkos::resize(scratch1Dviews[qnew], config_.nlev * aero_model::pcnst);
    Kokkos::resize(scratch1Dviews[dlfdp], config_.nlev);
  } // end(init)

  KOKKOS_INLINE_FUNCTION
  void compute_tendencies(const AeroConfig &config, const ThreadTeam &team,
                          Real t, Real dt, const Atmosphere &atmosphere,
                          const Prognostics &prognostics,
                          const Diagnostics &diagnostics,
                          const Tendencies &tendencies) const;
};

namespace convproc {
using Const_Kokkos_2D_View =
    Kokkos::View<const Real * [ConvProc::pcnst_extd], Kokkos::MemoryUnmanaged>;
using Kokkos_2D_View =
    Kokkos::View<Real * [ConvProc::pcnst_extd], Kokkos::MemoryUnmanaged>;

KOKKOS_INLINE_FUNCTION
void assign_la_lc(const int imode, const int ispec, int &la, int &lc) {
  // ---------------------------------------------------------------------
  // get the index of interstital (la) and cloudborne (lc) aerosols
  // from mode index and species index
  // Cloudborne aerosols are appended after interstitial aerosol array
  // so lc (cloudborne) is offset from ic (interstitial) by
  // aero_model::pcnst.
  //-----------------------------------------------------------------------
  if (ispec == -1) {
    la = ConvProc::numptr_amode(imode);
    lc = ConvProc::numptrcw_amode(imode);
  } else {
    la = ConvProc::lmassptr_amode(ispec, imode);
    lc = ConvProc::lmassptrcw_amode(ispec, imode);
  }
  lc += aero_model::pcnst;
}

// nsrflx is the number of process-specific column tracer tendencies:
// activation, resuspension, aqueous chemistry, wet removal, actual and pseudo.
static constexpr int nsrflx = 6;
// clang-format off
KOKKOS_INLINE_FUNCTION
void update_tendency_diagnostics(
    const int ntsub,   // IN  number of sub timesteps
    const int ncnst,   // IN  number of tracers to transport
    const bool doconvproc[], // IN  flag for doing convective transport
    Real sumactiva[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dconudt_activa [kg/kg/s * mb]
    Real sumaqchem[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dconudt_aqchem [kg/kg/s * mb]
    Real sumwetdep[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dconudt_wetdep [kg/kg/s * mb]
    Real sumresusp[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dcondt_resusp [kg/kg/s * mb]
    Real sumprevap[ConvProc::pcnst_extd], // INOUT sum (over layers) of dp*dcondt_prevap [kg/kg/s * mb]
    Real sumprevap_hist[ConvProc::pcnst_extd],// INOUT sum (over layers) of dp*dcondt_prevap_hist [kg/kg/s * mb]
    Real qsrflx[/* aero_model::pcnst */][nsrflx]) // INOUT process-specific column tracer tendencies [kg/m2/s]
{

  // -----------------------------------------------------------------------
  //  update tendencies to final output of ma_convproc_tend
  // 
  //  note that the ma_convproc_tend does not apply convective cloud processing
  //     to the stratiform-cloudborne aerosol
  //  within this routine, cloudborne aerosols are convective-cloudborne
  // 
  //  the individual process column tendencies (sumwetdep, sumprevap, ...)
  //     are just diagnostic fields that can be written to history
  //  tendencies for interstitial and convective-cloudborne aerosol could
  //     both be passed back and output, if desired
  //  currently, however, the interstitial and convective-cloudborne tendencies
  //     are combined (in the next code block) before being passed back (in qsrflx)
  // -----------------------------------------------------------------------

  // clang-format on

  EKAT_KERNEL_REQUIRE(aero_model::pcnst == ncnst);
  const int ntot_amode = AeroConfig::num_modes();
  int la = 0, lc = 0;
  // update diagnostic variables
  for (int imode = 0; imode < ntot_amode; ++imode) {
    for (int ispec = -1; ispec < num_species_mode(imode); ++ispec) {
      // cloudborne aerosols are appended after intersitial
      assign_la_lc(imode, ispec, la, lc);
      if (doconvproc[la]) {
        sumactiva[la] += sumactiva[lc];
        sumresusp[la] += sumresusp[lc];
        sumaqchem[la] += sumaqchem[lc];
        sumwetdep[la] += sumwetdep[lc];
        sumprevap[la] += sumprevap[lc];
        sumprevap_hist[la] += sumprevap_hist[lc];
      }
    }
  }
  // scatter overall tendency back to full array
  // The indexing started at 2 for Fortran, so 1 for C++
  const Real hund_ovr_g = 100.0 / Constants::gravity;
  const Real xinv_ntsub = 1.0 / ntsub;
  Real qsrflx_i[6] = {};
  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc[icnst]) {
      // scatter column burden tendencies for various processes to qsrflx
      // process-specific column tracer tendencies [kg/m2/s]
      //   0 = activation   of interstial to conv-cloudborne
      //   1 = resuspension of conv-cloudborne to interstital
      //   2 = aqueous chemistry (not implemented yet, so zero)
      //   3 = wet removal
      //   4 = actual precip-evap resuspension (what actually is applied to a
      //   species)
      //   5 = pseudo precip-evap resuspension (for history file)
      qsrflx_i[0] = sumactiva[icnst] * hund_ovr_g;
      qsrflx_i[1] = sumresusp[icnst] * hund_ovr_g;
      qsrflx_i[2] = sumaqchem[icnst] * hund_ovr_g;
      qsrflx_i[3] = sumwetdep[icnst] * hund_ovr_g;
      qsrflx_i[4] = sumprevap[icnst] * hund_ovr_g;
      qsrflx_i[5] = sumprevap_hist[icnst] * hund_ovr_g;
      for (int i = 0; i < nsrflx; ++i) {
        qsrflx[icnst][i] += qsrflx_i[i] * xinv_ntsub;
      }
    }
  }
}
// =========================================================================================
// clang-format off
template <typename SubView, typename ConstSubView>
KOKKOS_INLINE_FUNCTION
void update_tendency_final(
    const int ntsub,   // IN  number of sub timesteps
    const int jtsub,   // IN  index of sub timesteps from the outer loop
    const int ncnst,   // IN  number of tracers to transport
    const Real dt,     // IN delta t (model time increment) [s]
    const ConstSubView dcondt, // IN grid-average TMR tendency for current column  [kg/kg/s]
    const bool doconvproc[], // IN  flag for doing convective transport
    SubView dqdt, // INOUT Tracer tendency array
    SubView q_i)  // INOUT  q(icol,kk,icnst) at current icol
{

  // -----------------------------------------------------------------------
  //  update tendencies to final output of ma_convproc_tend
  // 
  //  note that the ma_convproc_tend does not apply convective cloud processing
  //     to the stratiform-cloudborne aerosol
  //  within this routine, cloudborne aerosols are convective-cloudborne
  // 
  //  before tendencies (dcondt, which is loaded into dqdt) are returned,
  //     the convective-cloudborne aerosol tendencies must be combined
  //     with the interstitial tendencies
  //  ma_resuspend_convproc has already done this for the dcondt
  // -----------------------------------------------------------------------

  // clang-format on
  // inverse of ntsub (1.0/ntsub)
  const Real xinv_ntsub = 1.0 / ntsub;
  // delta t of sub timestep (dt/ntsub) [s]
  const Real dtsub = dt * xinv_ntsub;

  // scatter overall tendency back to full array
  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc[icnst]) {
      // scatter overall dqdt tendency back
      const Real dqdt_i = dcondt[icnst];
      dqdt[icnst] += dqdt_i * xinv_ntsub;
      // update the q_i for the next interation of the jtsub loop
      if (jtsub < ntsub) {
        q_i[icnst] = mam4::max((q_i[icnst] + dqdt_i * dtsub), 0.0);
      }
    }
  }
}
// =========================================================================================
// clang-format off
// nlev = number of atmospheric levels: 0 <= ktop <= kbot_prevap <= nvel
// nlevp = nlev + 1
//
// (note:  TMR = tracer mixing ratio)
//
KOKKOS_INLINE_FUNCTION
void compute_column_tendency(
  const bool doconvproc_extd[ConvProc::pcnst_extd],  // IN flag for doing convective transport
  const int ktop,                                    // IN top level index
  const int kbot_prevap,                             // IN bottom level index, for resuspension and evaporation only
  const Real dpdry[/*nlev*/],                        // IN dp [mb]
  Const_Kokkos_2D_View dcondt_resusp,      // IN nlev - portion of TMR tendency due to resuspension [kg/kg/s]
  Const_Kokkos_2D_View dcondt_prevap,      // IN nlev - portion of TMR tendency due to precip evaporation [kg/kg/s]
  Const_Kokkos_2D_View dcondt_prevap_hist, // IN nlev - portion of TMR tendency due to precip evaporation, goes into the history [kg/kg/s]
  Const_Kokkos_2D_View dconudt_activa,     // IN nlevp- d(conu)/dt by activation [kg/kg/s]
  Const_Kokkos_2D_View dconudt_wetdep,     // IN nlevp- d(conu)/dt by wet removal[kg/kg/s]
  const Real fa_u[/*nlev*/],                       //  IN  fractional area of the updraft [fraction]
  Real sumactiva[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_activa [kg/kg/s * mb] 
  Real sumaqchem[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_aqchem [kg/kg/s * mb]
  Real sumwetdep[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_wetdep [kg/kg/s * mb]
  Real sumresusp[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_resusp [kg/kg/s * mb]
  Real sumprevap[ConvProc::pcnst_extd],  //  IN/OUT sum (over layers) of dp*dconudt_prevap [kg/kg/s * mb]
  Real sumprevap_hist[ConvProc::pcnst_extd]) // IN/OUT sum (over layers) of dp*dconudt_prevap_hist [kg/kg/s * mb]
{
  // clang-format on
  const Real dconudt_aqchem = 0; // aqueous chemistry is ignored in current code
  // initialize variables
  for (int i = 0; i < ConvProc::pcnst_extd; ++i) {
    sumactiva[i] = 0;
    sumaqchem[i] = 0;
    sumwetdep[i] = 0;
    sumresusp[i] = 0;
    sumprevap[i] = 0;
    sumprevap_hist[i] = 0;
  }

  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // should go to kk=nlev for dcondt_prevap, and this should be safe for
      // other sums
      for (int kk = ktop; kk < kbot_prevap; ++kk) {
        sumactiva[icnst] += dconudt_activa(kk, icnst) * dpdry[kk] * fa_u[kk];
        sumaqchem[icnst] += dconudt_aqchem * dpdry[kk] * fa_u[kk];
        sumwetdep[icnst] += dconudt_wetdep(kk, icnst) * dpdry[kk] * fa_u[kk];
        sumresusp[icnst] += dcondt_resusp(kk, icnst) * dpdry[kk];
        sumprevap[icnst] += dcondt_prevap(kk, icnst) * dpdry[kk];
        sumprevap_hist[icnst] += dcondt_prevap_hist(kk, icnst) * dpdry[kk];
      }
    }
  }
}
// =========================================================================================
template <typename SubView>
KOKKOS_INLINE_FUNCTION void tmr_tendency(const int la, const int lc,
                                         SubView dcondt,
                                         SubView dcondt_resusp) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  calculate tendency of TMR
  // -----------------------------------------------------------------------
  // arguments (note:  TMR = tracer mixing ratio)
  /*
    in    :: la, lc             indices from assign_la_lc
    inout :: dcondt[pcnst_extd]  overall TMR tendency from convection [#/kg/s or kg/kg/s] 
    inout :: dcondt_resusp[pcnst_extd] portion of TMR tendency due to resuspension [#/kg/s or kg/kg/s]
  */
  // Only apply adjustments to dcondt for pairs of 
  // unactivated (la) and activated (lc) aerosol species
  // clang-format on
  if (la > -1 && la < ConvProc::pcnst_extd && lc > -1 &&
      lc < ConvProc::pcnst_extd) {
    // cam5 approach
    dcondt[la] += dcondt[lc];
    dcondt_resusp[la] = dcondt[lc];
    dcondt_resusp[lc] = -dcondt[lc];
    dcondt[lc] = 0;
  }
}

// =========================================================================================
template <class SubView>
KOKKOS_INLINE_FUNCTION void ma_resuspend_convproc(SubView dcondt,
                                                  SubView dcondt_resusp) {
  // -----------------------------------------------------------------------
  //
  //  Purpose:
  //  Calculate resuspension of activated aerosol species resulting from both
  //     detrainment from updraft and downdraft into environment
  //     subsidence and lifting of environment, which may move air from
  //        levels with large-scale cloud to levels with no large-scale cloud
  //
  //  Method:
  //  Three possible approaches were considered:
  //
  //  1. Ad-hoc #1 approach.  At each level, adjust dcondt for the activated
  //     and unactivated portions of a particular aerosol species so that the
  //     ratio of dcondt (activated/unactivate) is equal to the ratio of the
  //     mixing ratios before convection.
  //     THIS WAS IMPLEMENTED IN MIRAGE2
  //
  //  2. Ad-hoc #2 approach.  At each level, adjust dcondt for the activated
  //     and unactivated portions of a particular aerosol species so that the
  //     change to the activated portion is minimized (zero if possible).  The
  //     would minimize effects of convection on the large-scale cloud.
  //     THIS IS CURRENTLY IMPLEMENTED IN CAM5 where we assume that convective
  //     clouds have no impact on the stratiform-cloudborne aerosol
  //
  //  3. Mechanistic approach that treats the details of interactions between
  //     the large-scale and convective clouds.  (Something for the future.)
  //
  //  Author: R. Easter
  //
  //  C++ porting: only method #2 is implemented.
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  //  arguments
  //  (note:  TMR = tracer mixing ratio)
  /*
    inout :: dcondt[pcnst_extd]
               overall TMR tendency from convection [#/kg/s or kg/kg/s]
    out   :: dcondt_resusp[pcnst_extd]
          portion of TMR tendency due to resuspension [#/kg/s or kg/kg/s]
          (actually, due to the adjustments made here)
  */
  for (int i = 0; i < ConvProc::pcnst_extd; ++i)
    dcondt_resusp[i] = 0;
  const int ntot_amode = AeroConfig::num_modes();
  int la, lc;
  for (int imode = 0; imode < ntot_amode; ++imode) {
    for (int ispec = -1; ispec < num_species_mode(imode); ++ispec) {
      // cloudborne aerosols are appended after intersitial
      assign_la_lc(imode, ispec, la, lc);
      tmr_tendency(la, lc, dcondt, dcondt_resusp);
    }
  }
}

// =========================================================================================
KOKKOS_INLINE_FUNCTION
void ma_precpevap(const Real dpdry, const Real evapc, const Real pr_flux,
                  Real &pr_flux_base, Real &pr_flux_tmp, Real &x_ratio) {
  // clang-format off
  // ------------------------------------------
  // step 1 in ma_precpevap_convproc: aerosol resuspension from precipitation evaporation
  // ------------------------------------------

  /*
  in      :: evapc         conv precipitataion evaporation rate [kg/kg/s]
  in      :: dpdry         pressure thickness of level [mb]
  in      :: pr_flux       precip flux at base of current layer [(kg/kg/s)*mb]
  inout   :: pr_flux_base  precip flux at an effective cloud base for calculations 
                           in a particular layer
  out     :: pr_flux_tmp   precip flux at base of current layer, after adjustment 
                           in step 1 [(kg/kg/s)*mb]
  out     :: x_ratio       ratio of adjusted and old fraction of precipitation-borne 
                           aerosol flux that is NOT resuspended, used in step 2
  */
  // clang-format on

  // a small value that variables smaller than it are considered as zero
  const Real small_value = 1.0e-30;

  // adjust pr_flux due to local evaporation
  const Real ev_flux_local = mam4::max(0.0, evapc * dpdry);
  pr_flux_tmp =
      utils::min_max_bound(0.0, pr_flux_base, pr_flux - ev_flux_local);

  x_ratio = 0.0;
  if (pr_flux_base < small_value) {
    // this will start things fresh at the next layer
    pr_flux_base = 0.0;
    pr_flux_tmp = 0.0;
    return;
  }

  // calculate fraction of resuspension
  Real pr_ratio_old = pr_flux / pr_flux_base;
  pr_ratio_old = utils::min_max_bound(0.0, 1.0, pr_ratio_old);
  // 2: log-normal distribution
  Real frac_aer_resusp_old =
      1.0 - WetDepTemp::faer_resusp_vs_fprec_evap_mpln(1.0 - pr_ratio_old, 2);
  frac_aer_resusp_old = utils::min_max_bound(0.0, 1.0, frac_aer_resusp_old);

  Real pr_ratio_tmp =
      utils::min_max_bound(0.0, 1.0, pr_flux_tmp / pr_flux_base);
  pr_ratio_tmp = mam4::min(pr_ratio_tmp, pr_ratio_old);
  // 2: log-normal distribution
  Real frac_aer_resusp_tmp =
      1.0 - WetDepTemp::faer_resusp_vs_fprec_evap_mpln(1.0 - pr_ratio_tmp, 2);
  frac_aer_resusp_tmp = utils::min_max_bound(0.0, 1.0, frac_aer_resusp_tmp);
  frac_aer_resusp_tmp = mam4::min(frac_aer_resusp_tmp, frac_aer_resusp_old);

  // compute x_ratio
  if (frac_aer_resusp_tmp > small_value) {
    x_ratio = frac_aer_resusp_tmp / frac_aer_resusp_old;
  } else {
    // this will start things fresh at the next layer
    pr_flux_base = 0.0;
    pr_flux_tmp = 0.0;
  }
}

//=========================================================================================
template <typename SubView, typename ConstSubView>
KOKKOS_INLINE_FUNCTION void
ma_precpprod(const Real rprd, const Real dpdry,
             const bool doconvproc_extd[ConvProc::pcnst_extd],
             const Real x_ratio, const int species_class[aero_model::pcnst],
             const int mmtoo_prevap_resusp[aero_model::pcnst], Real &pr_flux,
             Real &pr_flux_tmp, Real &pr_flux_base, ColumnView wd_flux,
             ConstSubView dcondt_wetdep, SubView dcondt, SubView dcondt_prevap,
             SubView dcondt_prevap_hist) {
  // clang-format off
  // ------------------------------------------
  //  step 2 in ma_precpevap_convproc: aerosol scavenging from precipitation production
  // ------------------------------------------
  /*
  in  rprd  -  conv precip production  rate (at a certain level) [kg/kg/s]
  in  dcondt_wetdep[pcnst_extd] - portion of TMR tendency due to wet removal [kg/kg/s]
  in  dpdry      - pressure thickness of level [mb]
  in  doconvproc_extd[pcnst_extd]  - indicates which species to process
  in  x_ratio    - ratio of adjusted and old fraction of precipitation-borne aerosol 
                   flux that is NOT resuspended, calculated in step 1
  in  species_class(:) specify what kind of species it is. defined as
          spec_class::undefined  = 0
          spec_class::cldphysics = 1
          spec_class::aerosol    = 2
          spec_class::gas        = 3
          spec_class::other      = 4
  in  mmtoo_prevap_resusp[aero_model::pcnst]
        pointers for resuspension mmtoo_prevap_resusp values are
           >=0 for aerosol mass species with    coarse mode counterpart
           -2 for aerosol mass species WITHOUT coarse mode counterpart
           -3 for aerosol number species
           -1 for other species

  inout  pr_flux   - precip flux at base of current layer [(kg/kg/s)*mb]
  inout  pr_flux_tmp   - precip flux at base of current layer, after adjustment in step 1 [(kg/kg/s)*mb]
  inout  pr_flux_base   - precip flux at an effective cloud base for calculations in a particular layer
  inout  wd_flux[pcnst_extd]   - tracer wet deposition flux at base of current layer [(kg/kg/s)*mb]
  inout  dcondt[pcnst_extd]  - overall TMR tendency from convection at a certain layer [kg/kg/s]
  inout  dcondt_prevap[pcnst_extd]  - portion of TMR tendency due to precip evaporation [kg/kg/s]
  inout  dcondt_prevap_hist[pcnst_extd]   - dcondt_prevap_hist at a certain layer [kg/kg/s]
  */
  // clang-format on
  // local precip flux [(kg/kg/s)*mb]
  const Real pr_flux_local = mam4::max(0.0, rprd * dpdry);
  pr_flux_base = mam4::max(0.0, pr_flux_base + pr_flux_local);
  pr_flux =
      utils::min_max_bound(0.0, pr_flux_base, pr_flux_tmp + pr_flux_local);

  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {

    if (doconvproc_extd[icnst]) {
      // wet deposition flux from the aerosol resuspension
      // wd_flux_tmp (updated) =
      //            (wd_flux coming into the layer) - (resuspension ! decrement)
      // wd_flux_tmp - updated wet deposition flux [(kg/kg/s)*mb]
      const Real wd_flux_tmp = mam4::max(0.0, wd_flux[icnst] * x_ratio);

      // change to wet deposition flux from evaporation [(kg/kg/s)*mb]
      const Real del_wd_flux_evap =
          mam4::max(0.0, wd_flux[icnst] - wd_flux_tmp);
      // wet deposition flux from the aerosol scavenging
      // wd_flux (updated) = (wd_flux after resuspension) - (scavenging !
      // increment)

      // local wet deposition flux [(kg/kg/s)*mb]
      const Real wd_flux_local = mam4::max(0.0, -dcondt_wetdep[icnst] * dpdry);
      wd_flux[icnst] = mam4::max(0.0, wd_flux_tmp + wd_flux_local);

      // dcondt due to wet deposition flux change [kg/kg/s]
      const Real dcondt_wdflux = del_wd_flux_evap / dpdry;

      // for interstitial icnst2=icnst;  for activated icnst2=icnst-pcnst
      const int icnst2 = icnst % aero_model::pcnst;

      // not sure what this mean exactly. Only do it for aerosol mass species
      // (mmtoo>0).  mmtoo<=0 represents aerosol number species
      const int mmtoo = mmtoo_prevap_resusp[icnst2];
      if (species_class[icnst2] == ConvProc::species_class::aerosol) {
        if (mmtoo >= 0) {
          // add the precip-evap (resuspension) to the history-tendency of the
          // current species
          dcondt_prevap_hist[icnst] += dcondt_wdflux;
          // add the precip-evap (resuspension) to the actual tendencies of
          // appropriate coarse-mode species
          dcondt_prevap[mmtoo] += dcondt_wdflux;
          dcondt[mmtoo] += dcondt_wdflux;
        }
      } else {
        // do this for trace gases (although currently modal_aero_convproc does
        // not treat trace gases)
        dcondt_prevap_hist[icnst] += dcondt_wdflux;
        dcondt_prevap[icnst] += dcondt_wdflux;
        dcondt[icnst] += dcondt_wdflux;
      }
    }
  }
}
// =========================================================================================
KOKKOS_INLINE_FUNCTION
void ma_precpevap_convproc(const int ktop, const int nlev,
                           Const_Kokkos_2D_View dcondt_wetdep,
                           const Real rprd[/* nlev */],
                           const Real evapc[/* nlev */],
                           const Real dpdry[/* nlev */],
                           const bool doconvproc_extd[ConvProc::pcnst_extd],
                           const int species_class[aero_model::pcnst],
                           const int mmtoo_prevap_resusp[aero_model::pcnst],
                           ColumnView wd_flux, Kokkos_2D_View dcondt_prevap,
                           Kokkos_2D_View dcondt_prevap_hist,
                           Kokkos_2D_View dcondt) {
  // clang-format off
  // -----------------------------------------------------------------------
  // 
  //  Purpose:
  //  Calculate resuspension of wet-removed aerosol species resulting precip evaporation
  // 
  //      for aerosol mass   species, do non-linear resuspension to coarse mode
  //      for aerosol number species, all the resuspension is done in wetdepa_v2, so do nothing here
  // 
  //  Author: R. Easter
  // 
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  //  arguments
  //  (note:  TMR = tracer mixing ratio)
  /*
    inout :: dcondt[nlev,pcnst_extd]
                 overall TMR tendency from convection [kg/kg/s]
    in    :: dcondt_wetdep[nlev,pcnst_extd]
                 portion of TMR tendency due to wet removal [kg/kg/s]
    inout :: dcondt_prevap[nlev,pcnst_extd]
                 portion of TMR tendency due to precip evaporation [kg/kg/s]
                 (actually, due to the adjustments made here)
                 (on entry, this is 0.0)
    inout :: dcondt_prevap_hist[nlev,pcnst_extd]
                 this determines what goes into the history
                    precip-evap SFSEC variables
                 currently, the SFSEC resuspension are attributed
                    to the species that got scavenged,
                    WHICH IS NOT the species that actually
                    receives the resuspension
                    when modal_aero_wetdep_resusp_opt > 0
                 so when scavenged so4_c1 is resuspended as so4_a1,
                    this resuspension column-tendency shows
                    up in so4_c1SFSES
                 this is done to allow better tracking of the
                    resuspension in the mass-budget post-processing s
    in    :: rprd   conv precip production  rate (gathered) [kg/kg/s]
    in    :: evapc  conv precip evaporation rate (gathered) [kg/kg/s]
    in    :: dpdry  pressure thickness of leve;
    in    :: doconvproc_extd[pcnst_extd]   indicates which species to process
    in    :: species_class[gas_pcnst]   specify what kind of species it is. defined at physconst.F90
                                   undefined  = 0
                                   cldphysics = 1
                                   aerosol    = 2
                                   gas        = 3
                                   other      = 4
    in  mmtoo_prevap_resusp[aero_model::pcnst]
         pointers for resuspension mmtoo_prevap_resusp values are
            >=0 for aerosol mass species with    coarse mode counterpart
            -2 for aerosol mass species WITHOUT coarse mode counterpart
            -3 for aerosol number species
            -1 for other species
  */
  //
  // *** note use of non-standard units
  //
  // precip
  //    dpdry is mb
  //    rprd and evapc are kgwtr/kgair/s
  //    pr_flux = dpdry(kk)*rprd is mb*kgwtr/kgair/s
  //
  // precip-borne aerosol
  //    dcondt_wetdep is kgaero/kgair/s
  //    wd_flux = tmpdp*dcondt_wetdep is mb*kgaero/kgair/s
  //    dcondt_prevap = del_wd_flux_evap/dpdry is kgaero/kgair/s
  // so this works ok too
  //
  // *** dilip switched from tmpdg (or dpdry) to tmpdpg = tmpdp/gravit
  // that is incorrect, but probably does not matter
  //    for aerosol, wd_flux units do not matter
  //        only important thing is that tmpdp (or tmpdpg) is used
  //        consistently when going from dcondt to wd_flux then to dcondt
  // clang-format on

  // initiate variables that are integrated in vertical
  // precip flux at base of current layer [(kg/kg/s)*mb]
  Real pr_flux = 0.0;
  // precip flux at an effective cloud base for calculations in a particular
  // layer
  Real pr_flux_base = 0.0;
  // precip flux at base of current layer, after adjustment of resuspension in
  // step 1 [(kg/kg/s)*mb]
  Real pr_flux_tmp = 0;
  // ratio of adjusted and old fraction of precipitation-borne aerosol
  // flux that is NOT resuspended, calculated in step 1 and used in step 2 (see
  // below)
  Real x_ratio = 0;

  // tracer wet deposition flux at base of current layer [(kg/kg/s)*mb]
  for (int i = 0; i < ConvProc::pcnst_extd; ++i)
    wd_flux[i] = 0;
  for (int kk = 0; kk < nlev; ++kk)
    for (int i = 0; i < ConvProc::pcnst_extd; ++i)
      dcondt_prevap(kk, i) = 0;
  for (int kk = 0; kk < nlev; ++kk)
    for (int i = 0; i < ConvProc::pcnst_extd; ++i)
      dcondt_prevap_hist(kk, i) = 0;

  for (int kk = ktop; kk < nlev; ++kk) {
    // step 1 - precip evaporation and aerosol resuspension
    ma_precpevap(dpdry[kk], evapc[kk], pr_flux, pr_flux_base, pr_flux_tmp,
                 x_ratio);
    // step 2 - precip production and aerosol scavenging
    auto dcondt_wetdep_sub = Kokkos::subview(dcondt_wetdep, kk, Kokkos::ALL());
    auto dcondt_sub = Kokkos::subview(dcondt, kk, Kokkos::ALL());
    auto dcondt_prevap_sub = Kokkos::subview(dcondt_prevap, kk, Kokkos::ALL());
    auto dcondt_prevap_hist_sub =
        Kokkos::subview(dcondt_prevap_hist, kk, Kokkos::ALL());
    ma_precpprod(rprd[kk], dpdry[kk], doconvproc_extd, x_ratio, species_class,
                 mmtoo_prevap_resusp, pr_flux, pr_flux_tmp, pr_flux_base,
                 wd_flux, dcondt_wetdep_sub, dcondt_sub, dcondt_prevap_sub,
                 dcondt_prevap_hist_sub);
  }
}

// =========================================================================================
// TODO: initialize_dcondt uses multiple levels for computation but ONLY
// sets a SINGLE level on output for the loop over ktop to kbot.  So, it should
// be possible to input kk and then call in parallel from ktop to kbot.
KOKKOS_INLINE_FUNCTION
void initialize_dcondt(const bool doconvproc_extd[ConvProc::pcnst_extd],
                       const int iflux_method, const int ktop, const int kbot,
                       const int nlev, const Real dpdry[/* nlev */],
                       const Real fa_u[/* nlev */], const Real mu[/* nlev+1 */],
                       const Real md[/* nlev+1 */], Const_Kokkos_2D_View chat,
                       Const_Kokkos_2D_View gath, Const_Kokkos_2D_View conu,
                       Const_Kokkos_2D_View cond,
                       Const_Kokkos_2D_View dconudt_activa,
                       Const_Kokkos_2D_View dconudt_wetdep,
                       const Real dudp[/* nlev */], const Real dddp[/* nlev */],
                       const Real eudp[/* nlev */], const Real eddp[/* nlev */],
                       Kokkos_2D_View dcondt) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  initialize dondt and update with aerosol activation and wetdeposition
  //  will update later with dcondt_prevap and dcondt_resusp
  //  NOTE:  The approach used in convtran applies to inert tracers and
  //         must be modified to include source and sink terms
  // -----------------------------------------------------------------------

  /* cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
   in :: doconvproc_extd[pcnst_extd] ! flag for doing convective transport
   in :: iflux_method             ! 1=as in convtran (deep), 2=uwsh
   in :: ktop                     ! top level index
   in :: kbot                     ! bottom level index
   in :: dpdry[nlev]              ! dp [mb]
   in :: fa_u[nlev]               ! fractional area of the updraft [fraction]
   in :: mu[nlev+1]               ! mu at current i (note nlev+1 dimension, see ma_convproc_tend) [mb/s]
   in :: md[nlev+1]               ! md at current i (note nlev+1 dimension) [mb/s]
   in :: chat[nlev+1][pcnst_extd]  ! mix ratio in env at interfaces [kg/kg]
   in :: gath[nlev][pcnst_extd]    ! gathered tracer array [kg/kg]
   in :: conu[nlev+1][pcnst_extd]  ! mix ratio in updraft at interfaces [kg/kg]
   in :: cond[nlev+1][pcnst_extd]  ! mix ratio in downdraft at interfaces [kg/kg]
   in :: dconudt_activa[nlev+1][pcnst_extd] ! d(conu)/dt by activation [kg/kg/s]
   in :: dconudt_wetdep[nlev+1][pcnst_extd] ! d(conu)/dt by wet removal[kg/kg/s]
   in :: dudp[nlev]           ! du[i][k]*dp[i][k] at current i [mb/s]
   in :: dddp[nlev]           ! dd[i][k]*dp[i][k] at current i [mb/s]
   in :: eudp[nlev]           ! eu[i][k]*dp[i][k] at current i [mb/s]
   in :: eddp[nlev]           ! ed[i][k]*dp[i][k] at current i [mb/s]
   out :: dcondt[nlev,pcnst_extd]  ! grid-average TMR tendency for current column  [kg/kg/s]
  */
  // clang-format on
  // initialize variables
  for (int i = 0; i < nlev; ++i)
    for (int j = 0; j < ConvProc::pcnst_extd; ++j)
      dcondt(i, j) = 0.;

  // loop from ktop to kbot
  for (int kk = ktop; kk < kbot; ++kk) {
    const int kp1 = kk + 1;
    const int kp1x = mam4::min(kp1, nlev - 1);
    const int km1x = mam4::max(kk - 1, 0);
    const Real fa_u_dp = fa_u[kk] * dpdry[kk];
    // The indexing started at 2 for Fortran, so 1 for C++
    for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
      if (doconvproc_extd[icnst]) {
        // compute fluxes as in convtran, and also source/sink terms
        // (version 3 limit fluxes outside convection to mass in appropriate
        // layer (these limiters are probably only safe for positive definite
        // quantitities (it assumes that mu and md already satify a courant
        // number limit of 1)
        Real fluxin = 0, fluxout = 0;
        if (iflux_method != 2) {
          fluxin = mu[kp1] * conu(kp1, icnst) +
                   mu[kk] * mam4::min(chat(kk, icnst), gath(km1x, icnst)) -
                   (md[kk] * cond(kk, icnst) +
                    md[kp1] * mam4::min(chat(kp1, icnst), gath(kp1x, icnst)));
          fluxout = mu[kk] * conu(kk, icnst) +
                    mu[kp1] * mam4::min(chat(kp1, icnst), gath(kk, icnst)) -
                    (md[kp1] * cond(kp1, icnst) +
                     md[kk] * mam4::min(chat(kk, icnst), gath(kk, icnst)));
        } else {
          // new method -- simple upstream method for the env subsidence
          // tmpa = net env mass flux (positive up) at top of layer k
          fluxin = mu[kp1] * conu(kp1, icnst) - md[kk] * cond(kk, icnst);
          fluxout = mu[kk] * conu(kk, icnst) - md[kp1] * cond(kp1, icnst);
          Real tmpa = -(mu[kk] + md[kk]);
          if (tmpa <= 0.0) {
            fluxin -= tmpa * gath(km1x, icnst);
          } else {
            fluxout += tmpa * gath(kk, icnst);
          }
          // tmpa = net env mass flux (positive up) at base of layer k
          tmpa = -(mu[kp1] + md[kp1]);
          if (tmpa >= 0.0) {
            fluxin += tmpa * gath(kp1x, icnst);
          } else {
            fluxout -= tmpa * gath(kk, icnst);
          }
        }
        //  net flux [kg/kg/s * mb]
        const Real netflux = fluxin - fluxout;

        // note for C++ refactoring:
        // I was trying to separate dconudt_activa and dconudt_wetdep out
        // into a subroutine, but for some reason it doesn't give consistent
        // dcondt values. have to leave them here.   Shuaiqi Tang, 2022
        const Real netsrce =
            fa_u_dp * (dconudt_activa(kk, icnst) + dconudt_wetdep(kk, icnst));
        dcondt(kk, icnst) = (netflux + netsrce) / dpdry[kk];
      }
    }
  }
}
// =========================================================================================
// TODO: compute_downdraft_mixing_ratio uses multiple levels for computation but
// ONLY sets a SINGLE level on output for the loop over ktop to kbot.  So, it
// should be possible to input kk and then call in parallel from ktop to kbot.
// It is a bit confusing that the level set is kk+1 so it would be better to
// rewrite as kkp1=>kk and kk=>kk-1 and iterate from ktop+1 to kbot inclusive.
KOKKOS_INLINE_FUNCTION
void compute_downdraft_mixing_ratio(
    const bool doconvproc_extd[ConvProc::pcnst_extd], const int ktop,
    const int kbot, const Real md_i[/* nlev+1 */], const Real eddp[/* nlev */],
    Const_Kokkos_2D_View gath, Kokkos_2D_View cond) {
  // clang-format off
  //----------------------------------------------------------------------
  // Compute downdraft mixing ratios from cloudtop to cloudbase
  // No special treatment is needed at k=2
  // No transformation or removal is applied in the downdraft
  // ---------------------------------------------------------------------

  /* cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
   in doconvproc_extd[pcnst_extd] ! flag for doing convective transport
   in ktop                     ! top level index
   in kbot                     ! bottom level index
   in md_i[nlev+1]              ! md at current i (note nlev+1 dimension) [mb/s]
   in eddp[nlev]               ! ed(i,k)*dp(i,k) at current i [mb/s]
   in gath[nlev,(pcnst_extd]  ! gathered tracer array [kg/kg]
   inout  cond[nlev+1,pcnst_extd,nlev+1] ! mix ratio in downdraft at interfaces [kg/kg]
  */
  // clang-format on
  // threshold below which we treat the mass fluxes as zero [mb/s]
  //  BAD_CONSTANT - used for both compute_downdraft_mixing_ratio and
  //  compute_massflux
  const Real mbsth = 1.e-15;
  for (int kk = ktop; kk < kbot; ++kk) {
    const int kp1 = kk + 1;
    // md_m_eddp = downdraft massflux at kp1, without detrainment between k,kp1
    const Real md_m_eddp = md_i[kk] - eddp[kk];
    if (md_m_eddp < -mbsth) {
      // The indexing started at 2 for Fortran, so 1 for C++
      for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
        if (doconvproc_extd[icnst]) {
          cond(kp1, icnst) =
              (md_i[kk] * cond(kk, icnst) - eddp[kk] * gath(kk, icnst)) /
              md_m_eddp;
        }
      }
    }
  }
}
// ==================================================================================
template <typename SubView>
KOKKOS_INLINE_FUNCTION void
update_conu_from_act_frac(SubView conu, SubView dconudt, const int la,
                          const int lc, const Real act_frac,
                          const Real dt_u_inv) {
  // clang-format off
  // ---------------------------------------------------------------------
  // update conu and dconudt from activation fraction
  // ---------------------------------------------------------------------
  /* arguments:
   inout :: conu[pcnst_extd]    ! TMR concentration [#/kg or kg/kg]
   inout :: dconudt[pcnst_extd] ! TMR tendencies due to activation [#/kg/s or kg/kg/s]
   in    :: act_frac            ! activation fraction [fraction]
   in    :: dt_u_inv            ! 1.0/dt_u  [1/s]
   in    :: la                  ! indices for interstitial aerosols
   in    :: lc                  ! indices for in-cloud water aerosols
  */
  // clang-format on
  const Real delact = utils::min_max_bound(0.0, conu[la], conu[la] * act_frac);
  // update conu in interstitial and in-cloud condition
  conu[la] -= delact;
  conu[lc] += delact;
  // update dconu/dt
  dconudt[la] = -delact * dt_u_inv;
  dconudt[lc] = delact * dt_u_inv;
}
template <typename SubView>
KOKKOS_INLINE_FUNCTION void
aer_vol_num_hygro(const AeroSpeciesView &aero_species, const SubView conu,
                  const Real rhoair, Real vaerosol[AeroConfig::num_modes()],
                  Real naerosol[AeroConfig::num_modes()],
                  Real hygro[AeroConfig::num_modes()]) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  calculate aerosol volume, number and hygroscopicity
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  //  arguments:
  /*
   conu = tracer mixing ratios in updraft at top of this (current) level. The
   conu are changed by activation
   in    :: conu[pcnst_extd]       ! TMR [#/kg or kg/kg]
   in    :: rhoair                 ! air density [kg/m3]
   out   :: vaerosol[ntot_amode]   ! int+act volume [m3/m3]
   out   :: naerosol[ntot_amode]   ! interstitial+activated number conc [#/m3]
   out   :: hygro[ntot_amode]      ! current hygroscopicity for int+act [unitless]
  */
  // clang-format on

  // a small value that variables smaller than it are considered as zero for
  // aerosol volume [m3/kg]
  const Real small_vol = 1.0e-35;

  // -----------------------------------------------------------------------
  const int ntot_amode = AeroConfig::num_modes();
  for (int imode = 0; imode < ntot_amode; ++imode) {
    // compute aerosol (or a+cw) volume and hygroscopicity
    Real tmp_vol = 0.0;
    Real tmp_hygro = 0.0;
    const int nspec_amode = mam4::num_species_mode(imode);
    for (int ispec = 0; ispec < nspec_amode; ++ispec) {
      // mass divided by density
      const Real tmp_vol_spec =
          mam4::max(conu[ConvProc::lmassptr_amode(ispec, imode)], 0.0) /
          ConvProc::specdens_amode(aero_species,
                                   ConvProc::lspectype_amode(ispec, imode));
      // total aerosol volume
      tmp_vol += tmp_vol_spec;
      //  volume*hygro suming up for all species
      tmp_hygro += tmp_vol_spec *
                   ConvProc::spechygro(aero_species,
                                       ConvProc::lspectype_amode(ispec, imode));
    }
    // change volume from m3/kgair to m3/m3air
    vaerosol[imode] = tmp_vol * rhoair;
    if (tmp_vol < small_vol) {
      hygro[imode] = 0.2;
    } else {
      hygro[imode] = tmp_hygro / tmp_vol;
    }

    // computer a (or a+cw) number and bound it
    const Real tmp_num = mam4::max(conu[ConvProc::numptr_amode(imode)], 0.0);
    const Real n_min = vaerosol[imode] * ConvProc::voltonumbhi_amode(imode);
    const Real n_max = vaerosol[imode] * ConvProc::voltonumblo_amode(imode);
    naerosol[imode] = utils::min_max_bound(n_min, n_max, tmp_num * rhoair);
  }
}
// ======================================================================================
// Note: compute_wup sets wup[kk] only so can be called in parallel over the
// vertical level index, kk.
KOKKOS_INLINE_FUNCTION
Real compute_wup(const int iconvtype, const Real mu_i_kk, const Real mu_i_kp1,
                 const Real cldfrac_i, const Real rhoair_i, const Real zmagl) {
  // -----------------------------------------------------------------------
  //  estimate updraft velocity (wup)
  //  do it differently for deep and shallow convection
  // -----------------------------------------------------------------------

  // clang-format off
  /*
  in :: iconvtype       ! 1=deep, 2=uw shallow
  in :: kk              ! vertical level index
  in :: mu_i_kk          ! mu at current i (note nlev+1 dimension) [mb/s]
  in :: mu_i_kp1         ! mu at current i+1 (note nlev+1 dimension) [mb/s]
  in :: cldfrac_i       ! cldfrac at current icol (with adjustments) [fraction]
  in :: rhoair_i        ! air density at current i [kg/m3]
  in :: zmagl           ! height above surface [m]
  out :: wup at kk    ! mean updraft vertical velocity at current level updraft [m/s]
  */
  // clang-format on
  // pre-defined minimum updraft [m/s]
  // Based on Lemone and Zipser (J Atmos Sci, 1980, p. 2455)
  const Real w_min = 0.1;
  // pre-defined peak updraft [m/s]
  // Lemone and Zipser (J Atmos Sci, 1980, p. 2455)
  const Real w_peak = 4.0;

  const Real hund_ovr_g = 100.0 / Constants::gravity;

  Real wup_kk = 0;
  if (iconvtype != 1) {
    // shallow - wup = (mup in kg/m2/s) / [rhoair * (updraft area)]
    wup_kk =
        (mu_i_kp1 + mu_i_kk) * 0.5 * hund_ovr_g / (rhoair_i * 0.5 * cldfrac_i);
    wup_kk = mam4::max(w_min, wup_kk);
  } else {
    // deep - the above method overestimates updraft area and underestimate wup
    // the following is based Lemone and Zipser (J Atmos Sci, 1980, p. 2455)
    // peak updraft (= 4 m/s) is sort of a "grand median" from their GATE data
    // and Thunderstorm Project data which they also show
    // the vertical profile shape is a crude fit to their median updraft profile
    // height above surface [km]
    const Real zkm = zmagl * 1.0e-3;
    if (1.0 <= zkm) {
      wup_kk = w_peak * mam4::pow((zkm / w_peak), 0.21);
    } else {
      wup_kk = 2.9897 * mam4::sqrt(zkm);
    }
    wup_kk = utils::min_max_bound(w_min, w_peak, wup_kk);
  }
  return wup_kk;
}
// ======================================================================================
KOKKOS_INLINE_FUNCTION
void compute_massflux(const int nlev, const int ktop, const int kbot,
                      const Real dpdry[/* nlev */], const Real du[/* nlev */],
                      const Real eu[/* nlev */], const Real ed[/* nlev */],
                      Real mu[/* nlev+1 */], Real md[/* nlev+1 */],
                      Real &xx_mfup_max) {
  // -----------------------------------------------------------------------
  //  compute dry mass fluxes
  //  This is approximate because the updraft air is has different temp and qv
  //  than the grid mean, but the whole convective parameterization is highly
  //  approximate values below a threshold and at "top of cloudtop", "base of
  //  cloudbase" are set as zero
  // -----------------------------------------------------------------------
  // clang-format off
  /*
   in  :: nlev           ! number of levels
   in  :: ktop           ! top level index
   in  :: kbot           ! bottom level index
   in  :: dpdry[nlev]    ! dp [mb]
   in  :: du[nlev]       ! Mass detrain rate from updraft [1/s]
   in  :: eu[nlev]       ! Mass entrain rate into updraft [1/s]
   in  :: ed[nlev]       ! Mass entrain rate into downdraft [1/s]
   out :: mu[nlev+1]     ! mu at current i (note nlev+1 dimension, see ma_convproc_tend) [mb/s]
   out :: md[nlev+1]     ! md at current i (note nlev+1 dimension) [mb/s]
   inout :: xx_mfup_max  ! diagnostic field of column maximum updraft mass flux [mb/s]
  */
  // clang-format on
  // threshold below which we treat the mass fluxes as zero [mb/s]
  //  BAD_CONSTANT - used for both compute_downdraft_mixing_ratio and
  //  compute_massflux
  const Real mbsth = 1.e-15;

  // load mass fluxes at cloud layers
  // then load mass fluxes only over cloud layers
  // excluding "top of cloudtop", "base of cloudbase"

  //  first calculate updraft and downdraft mass fluxes for all layers
  for (int i = 0; i < nlev + 1; ++i)
    mu[i] = 0, md[i] = 0;
  // (eu-du) = d(mu)/dp -- integrate upwards, multiplying by dpdry
  for (int kk = nlev - 1; 0 <= kk; --kk) {
    mu[kk] = mu[kk + 1] + (eu[kk] - du[kk]) * dpdry[kk];
    xx_mfup_max = mam4::max(xx_mfup_max, mu[kk]);
  }
  // (ed) = d(md)/dp -- integrate downwards, multiplying by dpdry
  for (int kk = 1; kk < nlev; ++kk)
    md[kk] = md[kk - 1] - ed[kk - 1] * dpdry[kk - 1];

  for (int kk = 0; kk < nlev + 1; ++kk) {
    if (ktop < kk && kk < kbot) {
      // zero out values below threshold
      if (mu[kk] <= mbsth)
        mu[kk] = 0;
      if (md[kk] >= -mbsth)
        md[kk] = 0;
    } else {
      mu[kk] = 0, md[kk] = 0;
    }
  }
}
// ======================================================================================
// Because the local variable courantmax is over the whole column, this can not
// be called in parallel.  Could pass courantmax back as an array and then
// max-reduc over it.
KOKKOS_INLINE_FUNCTION
void compute_ent_det_dp(const int nlev, const int ktop, const int kbot,
                        const Real dt, const Real dpdry[/* nlev */],
                        const Real mu[/* nlev+1 */],
                        const Real md[/* nlev+1 */], const Real du[/* nlev */],
                        const Real eu[/* nlev */], const Real ed[/* nlev */],
                        int &ntsub, Real eudp[/* nlev */],
                        Real dudp[/* nlev */], Real eddp[/* nlev */],
                        Real dddp[/* nlev */]) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  calculate mass flux change (from entrainment or detrainment) in the current dp
  //  also get number of time substeps
  // -----------------------------------------------------------------------
  /*
  in  :: ii             ! index to gathered arrays
  in  :: ktop           ! top level index
  in  :: kbot           ! bottom level index
  in  :: dt             ! delta t (model time increment) [s]
  in  :: dpdry[nlev]    ! dp [mb]
  in  :: mu[nlev+1]     ! mu at current i (note nlev+1 dimension, see ma_convproc_tend) [mb/s]
  in  :: md[nlev+1]     ! md at current i (note nlev+1 dimension) [mb/s]
  in  :: du[nlev]       ! Mass detrain rate from updraft [1/s]
  in  :: eu[nlev]       ! Mass entrain rate into updraft [1/s]
  in  :: ed[nlev]       ! Mass entrain rate into downdraft [1/s]

  out :: ntsub          ! number of sub timesteps
  out :: eudp[nlev]     ! eu[i][k]*dp[i][k] at current i [mb/s]
  out :: dudp[nlev]     ! du[i][k]*dp[i][k] at current i [mb/s]
  out :: eddp[nlev]     ! ed[i][k]*dp[i][k] at current i [mb/s]
  out :: dddp[nlev]     ! dd[i][k]*dp[i][k] at current i [mb/s]
  */
  // clang-format on

  for (int i = 0; i < nlev; ++i)
    eudp[i] = dudp[i] = eddp[i] = dddp[i] = 0.0;

  // maximum value of courant number [unitless]
  Real courantmax = 0.0;
  ntsub = 1;

  //  Compute updraft and downdraft "entrainment*dp" from eu and ed
  //  Compute "detrainment*dp" from mass conservation (total is mass flux
  //  difference between the top an bottom interface of this layer)
  for (int kk = ktop; kk < kbot; ++kk) {
    if ((mu[kk] > 0) || (mu[kk + 1] > 0)) {
      if (du[kk] <= 0.0) {
        eudp[kk] = mu[kk] - mu[kk + 1];
      } else {
        eudp[kk] = mam4::max(eu[kk] * dpdry[kk], 0.0);
        dudp[kk] = (mu[kk + 1] + eudp[kk]) - mu[kk];
        if (dudp[kk] < 1.0e-12 * eudp[kk]) {
          eudp[kk] = mu[kk] - mu[kk + 1];
          dudp[kk] = 0.0;
        }
      }
    }
    if ((md[kk] < 0) || (md[kk + 1] < 0)) {
      eddp[kk] = mam4::max(ed[kk] * dpdry[kk], 0.0);
      dddp[kk] = (md[kk + 1] + eddp[kk]) - md[kk];
      if (dddp[kk] < 1.0e-12 * eddp[kk]) {
        eddp[kk] = md[kk] - md[kk + 1];
        dddp[kk] = 0.0;
      }
    }
    // get courantmax to calculate ntsub
    courantmax =
        mam4::max(courantmax,
                  (mu[kk + 1] + eudp[kk] - md[kk] + eddp[kk]) * dt / dpdry[kk]);
  }
  // number of time substeps needed to maintain "courant number" <= 1
  if (courantmax > (1.0 + 1.0e-6)) {
    ntsub = 1 + static_cast<int>(courantmax);
  }
}
// ======================================================================================
// This function can not be called in parallel over kk,
// it is a recursive calculation by level
KOKKOS_INLINE_FUNCTION
void compute_midlev_height(const int nlev, const Real dpdry[/* nlev */],
                           const Real rhoair[/* nlev */],
                           Real zmagl[/* nlev */]) {
  // -----------------------------------------------------------------------
  //  compute height above surface for middle of level kk
  // -----------------------------------------------------------------------
  /*
  in  :: dpdry[nlev]  ! dp [mb]
  in  :: rhoair[nlev] ! air density [kg/m3]
  out :: zmagl[nlev]    ! height above surface at middle level [m]
  */

  const Real hund_ovr_g = 100.0 / Constants::gravity;
  const int surface = nlev - 1;
  for (int i = 0; i < nlev; ++i)
    zmagl[i] = 0;
  // at surface layer thickness [m]
  Real dz = dpdry[surface] * hund_ovr_g / rhoair[surface];
  zmagl[surface] = 0.5 * dz;
  // other levels
  for (int kk = surface - 1; 0 <= kk; --kk) {
    // add half layer below
    zmagl[kk] = zmagl[kk + 1] + 0.5 * dz;

    // update layer thickness at level kk
    dz = dpdry[kk] * hund_ovr_g / rhoair[kk];

    // add half layer in this level
    zmagl[kk] += 0.5 * dz;
  }
}

// ======================================================================================
// I think if gath were set to q_i before calling this function and then a const
// gath was passed instead of a const q_i, then this function could take a kk
// and be called in parallel over the levels.
KOKKOS_INLINE_FUNCTION
void initialize_tmr_array(
    const int nlev, const int iconvtype,
    const bool doconvproc_extd[ConvProc::pcnst_extd],
    Kokkos::View<Real * [aero_model::pcnst], Kokkos::MemoryUnmanaged> q,
    Kokkos_2D_View gath, Kokkos_2D_View chat, Kokkos_2D_View conu,
    Kokkos_2D_View cond) {
  // -----------------------------------------------------------------------
  //  initialize tracer mixing ratio arrays (const, chat, conu, cond)
  //  chat, conu and cond are at interfaces; interpolation needed
  //  Note: for deep convection, some values between the two layers
  //  differ significantly, use geometric averaging under certain conditions
  // -----------------------------------------------------------------------

  // clang-format off
  /*
  in :: iconvtype                 ! 1=deep, 2=uw shallow
  in :: doconvproc_extd[pcnst_extd] ! flag for doing convective transport
  in :: q[nlev][pcnst]          ! q[icol][kk][icnst] at current icol

  out :: gath[nlev  ][pcnst_extd]   ! gathered tracer array [kg/kg]
  out :: chat[nlev+1][pcnst_extd]   ! mix ratio in env at interfaces [kg/kg]
  out :: conu[nlev+1][pcnst_extd]   ! mix ratio in updraft at interfaces [kg/kg]
  out :: cond[nlev+1][pcnst_extd]   ! mix ratio in downdraft at interfaces [kg/kg]
  */
  // clang-format on

  // threshold of constitute as zero [kg/kg]
  const Real small_con = 1.e-36;
  // small value for relative comparison
  // BAD_CONSTANT - is there a reference for this value?
  const Real small_rel = 1.0e-6;

  const int ncnst = aero_model::pcnst;
  const int pcnst_extd = ConvProc::pcnst_extd;

  // initiate variables
  for (int j = 0; j < nlev; ++j)
    for (int i = 0; i < pcnst_extd; ++i)
      gath(j, i) = 0;
  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // Gather up the constituent
      for (int kk = 0; kk < nlev; ++kk)
        gath(kk, icnst) = q(kk, icnst);
    }
  }

  for (int j = 0; j < nlev + 1; ++j)
    for (int i = 0; i < pcnst_extd; ++i)
      chat(j, i) = conu(j, i) = cond(j, i) = 0;

  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // Interpolate environment tracer values to interfaces
      for (int kk = 0; kk < nlev; ++kk) {
        const int km1 = mam4::max(0, kk - 1);
        // get relative difference between the two levels

        // min gath concentration at level kk and kk-1 [kg/kg]
        const Real min_con = mam4::min(gath(km1, icnst), gath(kk, icnst));
        // max gath concentration at level kk and kk-1 [kg/kg]
        const Real max_con = mam4::max(gath(km1, icnst), gath(kk, icnst));

        // relative difference between level kk and kk-1 [unitless]
        const Real c_dif_rel =
            min_con < 0 ? 0
                        : mam4::abs(gath(kk, icnst) - gath(km1, icnst)) /
                              mam4::max(max_con, small_con);

        // If the two layers differ significantly use a geometric averaging
        // procedure But only do that for deep convection.  For shallow, use the
        // simple averaging which is used in subr cmfmca
        if (iconvtype != 1) {
          // simple averaging for non-deep convection
          chat(kk, icnst) = 0.5 * (gath(kk, icnst) + gath(km1, icnst));
        } else if (c_dif_rel > small_rel) {
          // deep convection using geometric averaging

          // gath at the above (kk-1 level) [kg/kg]
          const Real c_above = mam4::max(gath(km1, icnst), max_con * 1.e-12);

          // gath at the below (kk level) [kg/kg]
          const Real c_below = mam4::max(gath(kk, icnst), max_con * 1.e-12);
          chat(kk, icnst) = mam4::log(c_above / c_below) / (c_above - c_below) *
                            c_above * c_below;
        } else {
          // Small diff, so just arithmetic mean
          chat(kk, icnst) = 0.5 * (gath(kk, icnst) + gath(kk, icnst));
        }
        // Set provisional up and down draft values, and tendencies
        conu(kk, icnst) = chat(kk, icnst);
        cond(kk, icnst) = chat(kk, icnst);
      }
    }
  }
  // The indexing started at 2 for Fortran, so 1 for C++
  for (int icnst = 1; icnst < ncnst; ++icnst) {
    if (doconvproc_extd[icnst]) {
      // Values at surface inferface == values in lowest layer
      chat(nlev, icnst) = gath(nlev - 1, icnst);
      conu(nlev, icnst) = gath(nlev - 1, icnst);
      cond(nlev, icnst) = gath(nlev - 1, icnst);
    }
  }
}

// ======================================================================================
KOKKOS_INLINE_FUNCTION
void set_cloudborne_vars(const bool doconvproc[aero_model::pcnst],
                         Real aqfrac[ConvProc::pcnst_extd],
                         bool doconvproc_extd[ConvProc::pcnst_extd]) {
  // -----------------------------------------------------------------------
  //  set cloudborne aerosol related variables:
  //  doconvproc_extd: extended array for both activated and unactivated
  //  aerosols aqfrac: set as 1.0 for activated aerosols and 0.0 otherwise
  // -----------------------------------------------------------------------
  /*
  // cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
  in :: doconvproc[pcnst]    ! flag for doing convective transport
  out :: doconvproc_extd[pcnst_extd]    ! flag for doing convective transport
  out :: aqfrac[pcnst_extd]  ! aqueous fraction of constituent in updraft
  [fraction]
  */
  const int pcnst_extd = ConvProc::pcnst_extd;
  const int gas_pcnst = aero_model::pcnst;
  const int num_modes = AeroConfig::num_modes();
  int la, lc;

  for (int i = 0; i < pcnst_extd; ++i)
    doconvproc_extd[i] = false;

  for (int i = 1; i < gas_pcnst; ++i)
    doconvproc_extd[i] = doconvproc[i];

  for (int i = 0; i < pcnst_extd; ++i)
    aqfrac[i] = 0;

  for (int imode = 0; imode < num_modes; ++imode) {
    const int nspec_amode = mam4::num_species_mode(imode);
    for (int ispec = -1; ispec < nspec_amode; ++ispec) {
      // append cloudborne aerosols after intersitial
      assign_la_lc(imode, ispec, la, lc);
      if (doconvproc[la]) {
        doconvproc_extd[lc] = true;
        aqfrac[lc] = 1.0;
      }
    }
  }
}
// ======================================================================================
template <typename SubView, typename ConstSubView>
KOKKOS_INLINE_FUNCTION void
update_qnew_ptend(const bool dotend[aero_model::pcnst],
                  const bool is_update_ptend, ConstSubView dqdt, const Real dt,
                  bool ptend_lq[aero_model::pcnst], SubView ptend_q,
                  SubView qnew) {
  // ---------------------------------------------------------------------------------------
  // update qnew, ptend_q and ptend_lq
  // ---------------------------------------------------------------------------------------

  // Arguments
  // clang-format off
  /*
   in :: dotend[pcnst]     ! if do tendency
   in :: is_update_ptend   ! if update ptend with dqdt
   in :: dqdt[pcnst] ! time tendency of tracer [kg/kg/s]
   in :: dt                ! model time step [s]
   inout :: ptend_lq[pcnst]  ! if do tendency
   inout :: ptend_q[pcnst] ! time tendency of q [kg/kg/s]
   inout :: qnew[pcnst]    ! Tracer array including moisture [kg/kg]
  */
  // clang-format on 
  for (int ll = 0; ll < aero_model::pcnst; ++ll) {
    // calc new q (after convproc tendency)
    if (dotend[ll]) qnew[ll] = mam4::max(0.0, qnew[ll] + dt*dqdt[ll]);
  }
  for (int ll = 0; ll < aero_model::pcnst; ++ll) {
    if (dotend[ll] && is_update_ptend) {
      // add dqdt onto ptend_q and set ptend_lq
      ptend_lq[ll] = true;
      ptend_q[ll] += dqdt[ll];
    }
  }
}
// ======================================================================================
// This can be parallelized over kk. All the "nlev" dimensioned arrays could be subscripted 
// with kk and passed as scalars or 1D arrays.
template <typename SubView>
KOKKOS_INLINE_FUNCTION
void compute_wetdep_tend(
  const bool doconvproc_extd[ConvProc::pcnst_extd],
  const Real dt,   
  const Real dt_u,   
  const Real dp,   
  const Real cldfrac_i,      
  const Real mu_p_eudp,   
  const Real aqfrac[ConvProc::pcnst_extd],         
  const Real icwmr,          
  const Real rprd,       
  SubView conu,           
  SubView dconudt_wetdep)
{
  // clang-format off
  // -----------------------------------------------------------------------
  //  compute tendency from wet deposition
  // 
  //     rprd               = precip formation as a grid-cell average (kgW/kgA/s)
  //     icwmr              = cloud water MR within updraft area (kgW/kgA)
  //     fupdr              = updraft fractional area (--)
  //     A = rprd/fupdr     = precip formation rate within updraft area (kgW/kgA/s)
  //     clw_preloss = cloud water MR before loss to precip
  //                 = icwmr + dt*(rprd/fupdr)
  //     B = A/clw_preloss  = (rprd/fupdr)/(icwmr + dt*rprd/fupdr)
  //                        = rprd/(fupdr*icwmr + dt*rprd)
  //                        = first-order removal rate (1/s)
  //     C = dp/(mup/fupdr) = updraft air residence time in the layer (s)
  // 
  //     fraction removed = (1.0 - exp(-cdt)) where
  //                  cdt = B*C = (fupdr*dp/mup)*[rprd/(fupdr*icwmr + dt*rprd)]
  // 
  //     Note1:  *** cdt is now sensitive to fupdr, which we do not really know,
  //                 and is not the same as the convective cloud fraction
  //     Note2:  dt is appropriate in the above cdt expression, not dtsub
  // 
  //     Apply wet removal at levels where
  //        icwmr(k) > clw_cut  AND  rprd(k) > 0.0
  //     as wet removal occurs in both liquid and ice clouds
  // -----------------------------------------------------------------------
  /*
   cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
   in :: doconvproc_extd[pcnst_extd] ! flag for doing convective transport
   in :: dt                   ! Model timestep [s]
   in :: dt_u                 ! lagrangian transport time in the updraft[s]
   in :: dp                  ! dp [mb]
   in :: cldfrac_i            ! cldfrac at current (with adjustments) [fraction]
   in :: mu_p_eudp            ! = mu_i[kp1] + eudp[k] [mb/s]
   in :: aqfrac[pcnst_extd]   ! aqueous fraction of constituent in updraft [fraction]
   in :: icwmr          ! Convective cloud water from zm scheme [kg/kg]
   in :: rprd           ! Convective precipitation formation rate [kg/kg/s]
   inout :: conu[pcnst_extd]   ! mix ratio in updraft at interfaces [kg/kg]
   inout :: dconudt_wetdep[pcnst_extd] ! d(conu)/dt by wet removal[kg/kg/s]
  */
  // clang-format on
  // cutoff value of cloud water for doing updraft [kg/kg]
  // BAD_CONSTANT - is there a reference for this value? and why is it
  // the same as small_rel as defined above?
  const Real clw_cut = 1.0e-6;

  // (in-updraft first order wet removal rate) * dt [unitless]
  Real cdt = 0.0;
  if (icwmr > clw_cut && rprd > 0.0) {
    const Real half_cld = 0.5 * cldfrac_i;
    cdt = (half_cld * dp / mu_p_eudp) * rprd / (half_cld * icwmr + dt * rprd);
  }
  if (cdt > 0.0) {
    const Real expcdtm1 = mam4::exp(-cdt) - 1;
    // The indexing started at 2 for Fortran, so 1 for C++
    for (int icnst = 1; icnst < ConvProc::pcnst_extd; ++icnst) {
      if (doconvproc_extd[icnst]) {
        dconudt_wetdep[icnst] = conu[icnst] * aqfrac[icnst] * expcdtm1;
        conu[icnst] += dconudt_wetdep[icnst];
        dconudt_wetdep[icnst] /= dt_u;
      }
    }
  }
}
// ======================================================================================
KOKKOS_INLINE_FUNCTION
void assign_dotend(const int species_class[aero_model::pcnst],
                   const bool convproc_do_aer, // true by default
                   const bool convproc_do_gas, // false by default
                   bool dotend[aero_model::pcnst]) {
  // ---------------------------------------------------------------------
  //  assign do-tendency flag from species_class, convproc_do_aer and
  //  convproc_do_gas. convproc_do_aer and convproc_do_gas are assigned in the
  //  beginning of the  module
  // ---------------------------------------------------------------------
  /*
  in    :: species_class[aero_model::pcnst]
  out   :: dotend[aero_model::pcnst]
  */
  //  turn on/off calculations for aerosols and trace gases
  for (int ll = 0; ll < aero_model::pcnst; ++ll) {
    if (species_class[ll] == ConvProc::species_class::aerosol &&
        convproc_do_aer) {
      dotend[ll] = true;
    } else if (species_class[ll] == ConvProc::species_class::gas &&
               convproc_do_gas) {
      dotend[ll] = true;
    } else {
      dotend[ll] = false;
    }
  }
}

// ======================================================================================
// This function just uses kk==kactfirst so can be called in parallel over kk.
template <typename SubView>
KOKKOS_INLINE_FUNCTION void
ma_activate_convproc(const AeroSpeciesView &aero_species, SubView conu,
                     SubView dconudt, const Real f_ent, const Real dt_u,
                     const Real wup, const Real tair, const Real rhoair,
                     const int kk, const int kactfirst) {
  // clang-format off
  // -----------------------------------------------------------------------
  // 
  //  Purpose:
  //  Calculate activation of aerosol species in convective updraft
  //  for a single column and level
  // 
  //  Method:
  //  conu(l)    = Updraft TMR (tracer mixing ratio) at k/k-1 interface
  //  f_ent      = Fraction of the "before-detrainment" updraft massflux at
  //               k/k-1 interface" resulting from entrainment of level k air
  //               (where k is the current level in subr ma_convproc_tend)
  // 
  //  On entry to this routine, the conu(l) represents the updraft TMR
  //  after entrainment, but before chemistry/physics and detrainment.
  // 
  //  This routine applies aerosol activation to the conu tracer mixing ratios,
  //  then adjusts the conu so that on exit,
  //    conu(la) = conu_incoming(la) - conu(la)*f_act(la)
  //    conu(lc) = conu_incoming(lc) + conu(la)*f_act(la)
  //  where
  //    la, lc   = indices for an unactivated/activated aerosol component pair
  //    f_act    = fraction of conu(la) that is activated.  The f_act are
  //               calculated with the Razzak-Ghan activation parameterization.
  //               The f_act differ for each mode, and for number/surface/mass.
  // 
  //  At cloud base (k==kactfirst), primary activation is done using the
  //  "standard" code in subr activate do diagnose maximum supersaturation.
  //  Above cloud base, secondary activation is done using a
  //  prescribed supersaturation.
  // 
  //  *** The updraft velocity used for activation calculations is rather
  //      uncertain and needs more work.  However, an updraft of 1-3 m/s
  //      will activate essentially all of accumulation and coarse mode particles.
  // 
  //  Author: R. Easter
  // 
  // -----------------------------------------------------------------------
  /*
   arguments  (note:  TMR = tracer mixing ratio)
   conu = tracer mixing ratios in updraft at top of this (current) level. The conu are changed by activation
   inout :: conu[pcnst_extd]    ! [#/kg or kg/kg]
   inout :: dconudt[pcnst_extd) ! TMR tendencies due to activation [#/kg/s or kg/kg/s]
   in    :: f_ent     ! fraction of updraft massflux that was entrained across this layer == eudp/mu_p_eudp [fraction]
   in    :: dt_u      ! lagrangian transport time in the updraft at current level [s]
   in    :: wup       ! mean updraft vertical velocity at current level updraft [m/s]
   in    :: tair      ! Temperature [K]
   in    :: rhoair    ! air density [kg/m3]
   in    :: kk        ! level index
   in    :: kactfirst ! k at cloud base
  */
  // clang-format on
  // check f_ent > 0
  if (f_ent <= 0.0)
    return;

  const int ntot_amode = AeroConfig::num_modes();

  Real exp45logsig[ntot_amode];
  Real alogsig[ntot_amode];
  Real aten;
  Real num2vol_ratio_min_nmodes[ntot_amode];
  Real num2vol_ratio_max_nmodes[ntot_amode];
  ndrop_od::ndrop_int(exp45logsig, alogsig, aten, num2vol_ratio_min_nmodes,
                      num2vol_ratio_max_nmodes);
  // The uniform or peak supersat value (as 0-1 fraction = percent*0.01)
  // BAD_CONSTANT - need a reference for this.
  const Real activate_smaxmax = 0.003;
  // int+act volume [m3/m3]
  Real vaerosol[ntot_amode] = {};
  // interstitial+activated number conc [#/m3]
  Real naerosol[ntot_amode] = {};
  // current hygroscopicity for int+act [unitless]
  Real hygro[ntot_amode] = {};

  // output of activate_modal not used in this subroutine. to understand this,
  // see subr activate_modal
  Real fluxm[ntot_amode] = {};
  // output of activate_modal not used in this subroutine. to understand this,
  // see subr activate_modal
  Real fluxn[ntot_amode] = {};
  // output of activate_modal not used in this subroutine. to understand this,
  // see subr activate_modal
  Real flux_fullact = 0;

  // mass fraction of aerosols activated [fraction]
  Real fm[ntot_amode] = {};
  // number fraction of aerosols activated [fraction]
  Real fn[ntot_amode] = {};
  //  calculate aerosol (or a+cw) volume, number and hygroscopicity
  aer_vol_num_hygro(aero_species, conu, rhoair, vaerosol, naerosol, hygro);

  //  call Razzak-Ghan activation routine with single updraft

  // mean updraft velocity [m/s]
  // force wbar >= 0.5 m/s for now
  const Real wbar = mam4::max(wup, 0.5);
  // limit for integration over updraft spectrum [m/s]
  Real wmaxf = wbar;

  if (kk == kactfirst) {
    // at cloud base - do primary activation
    ndrop_od::activate_modal(wbar, wmaxf, tair, rhoair, naerosol, vaerosol,
                             hygro, exp45logsig, alogsig, aten, fn, fm, fluxn,
                             fluxm, flux_fullact);
  } else {
    // above cloud base - do secondary activation with prescribed supersat
    // that is constant with height
    ndrop_od::activate_modal(wbar, wmaxf, tair, rhoair, naerosol, vaerosol,
                             hygro, exp45logsig, alogsig, aten, fn, fm, fluxn,
                             fluxm, flux_fullact, activate_smaxmax);
  }

  //  apply the activation fractions to the updraft aerosol mixing ratios

  // 1.0/dt_u [1/s]
  const Real dt_u_inv = 1.0 / dt_u;

  for (int imode = 0; imode < ntot_amode; ++imode) {
    int la, lc;
    // for aerosol number
    // cloudborne aerosols are appended after intersitial
    assign_la_lc(imode, -1, la, lc);
    //  activation fraction [fraction]
    const Real act_frac = fn[imode];
    update_conu_from_act_frac(conu, dconudt, la, lc, act_frac, dt_u_inv);
    // for aerosol mass
    const int nspec_amode = mam4::num_species_mode(imode);
    for (int ispec = 0; ispec < nspec_amode; ++ispec) {
      // cloudborne aerosols are appended after intersitial
      assign_la_lc(imode, ispec, la, lc);
      const Real act_frac = fm[imode];
      update_conu_from_act_frac(conu, dconudt, la, lc, act_frac, dt_u_inv);
    }
  }
}

// ======================================================================================
// Can be called in parallel over kk, not used for subscripting.
template <typename SubView>
KOKKOS_INLINE_FUNCTION void compute_activation_tend(
    const AeroSpeciesView &aero_species, const Real f_ent, const Real cldfrac,
    const Real rhoair, const Real mu_i_kk, const Real mu_i_kp1, const Real dt_u,
    const Real wup, const Real icwmr, const Real temperature, const int kk,
    int &kactcnt, int &kactfirst, SubView conu, SubView dconudt_activa,
    Real &xx_wcldbase, int &xx_kcldbase) {
  // clang-format off
  // -----------------------------------------------------------------------
  //  aerosol activation - method 2
  //     when kactcnt=1 (first/lowest layer with cloud water)
  //        apply "primary" activatation to the entire updraft
  //     when kactcnt>1
  //        apply secondary activatation to the entire updraft
  //        do this for all levels above cloud base (even if completely glaciated)
  //           (this is something for sensitivity testing)
  // -----------------------------------------------------------------------
  /*
    cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
   in :: kk                   ! vertical level index
   in :: f_ent                ! fraction of updraft massflux that was entrained across this layer == eudp/mu_p_eudp [fraction]
   in :: cldfrac         ! cldfrac at current icol (with adjustments) [fraction]
   in :: rhoair             ! air density at current i [kg/m3]
   in :: mu_i_kk mu_i at kk    ! mu at current i (note nlev+1 dimension) [mb/s]
   in :: mu_i_kp1 mu_i at kk+1 ! mu at current i (note nlev+1 dimension) [mb/s]
   in :: dt_u dt_u at kk   ! lagrangian transport time in the updraft at current level [s]
   in :: wup  wup at kk     ! mean updraft vertical velocity at current level updraft [m/s]
   in :: temperature           ! Temperature [K]
   in :: icwmr          ! Convective cloud water from zm scheme [kg/kg]
   inout :: kactcnt           !  Counter for no. of levels having activation
   inout :: kactfirst         ! Lowest layer with activation (= cloudbase)
   inout :: conu[pcnst_extd]   ! mix ratio in updraft at interfaces [kg/kg]
   inout :: dconudt_activa[pcnst_extd] ! d(conu)/dt by activation [kg/kg/s]
   inout :: xx_wcldbase ! w at first cloudy layer [m/s]
   inout :: xx_kcldbase ! level of cloud base
            xx_kcldbase is filled with index kk. maybe better define as integer
  // clang-format on
  */
  // cutoff value of cloud water for doing updraft [kg/kg]
  // BAD_CONSTANT - is there a reference for this value? and why is it
  // the same as small_rel as defined above?
  const Real clw_cut = 1.0e-6;
  const Real hund_ovr_g = 100.0 / Constants::gravity;
  // flag for doing activation at current level
  bool do_act_this_lev = false;
  if (kactcnt <= 0) {
    if (icwmr > clw_cut) {
      do_act_this_lev = true;
      kactcnt = 1;
      kactfirst = kk;
      // diagnostic fields
      // xx_wcldbase = w at first cloudy layer, estimated from mu and cldfrac
      xx_wcldbase = (mu_i_kp1 + mu_i_kk)*0.5*hund_ovr_g / (rhoair * (cldfrac*0.5));
      xx_kcldbase = kk;
    }
  } else {
    do_act_this_lev = true;
    ++kactcnt;
  }

  if ( do_act_this_lev ) {
    ma_activate_convproc(aero_species,
      conu, dconudt_activa, f_ent, dt_u, wup,
      temperature, rhoair, kk,  kactfirst);
  }
}
// ======================================================================================
// The loop over kk in this function can not be done in parallel as it it a recursive
// calculation of conu from one level to the next.
KOKKOS_INLINE_FUNCTION
void compute_updraft_mixing_ratio(
  const AeroSpeciesView &aero_species,
  const bool doconvproc_extd[ConvProc::pcnst_extd],
  const int nlev,
  const int ktop,   
  const int kbot,   
  const int iconvtype, 
  const Real dt,     
  const Real dp[/* nlev */],
  const Real dpdry[/* nlev */],        
  const Real cldfrac[/* nlev */],          
  const Real rhoair[/* nlev */],       
  const Real zmagl[/* nlev */],  
  const Real dz,     
  const Real mu[/* nlev+1 */],   
  const Real eudp[/* nlev */],    
  Const_Kokkos_2D_View gath,
  const Real temperature[/* nlev */],    
  const Real aqfrac[ConvProc::pcnst_extd], 
  const Real icwmr[/* nlev */],  
  const Real rprd[/* nlev */],   
  Real fa_u[/* nlev */],   
  Kokkos_2D_View dconudt_wetdep,
  Kokkos_2D_View dconudt_activa,
  Kokkos_2D_View conu,           
  Real &xx_wcldbase,    
  int &xx_kcldbase)
{
  // clang-format off
  // -----------------------------------------------------------------------
  //  Compute updraft mixing ratios from cloudbase to cloudtop
  //  as well as tendencies of wetdeposition and activation from updraft
  //  No special treatment is needed at k=nlev because arrays are dimensioned 1:nlev+1
  //  A time-split approach is used.  First, entrainment is applied to produce
  //     an initial conu(m,k) from conu(m,k+1).  Next, chemistry/physics are
  //     applied to the initial conu(m,k) to produce a final conu(m,k).
  //     Detrainment from the updraft uses this final conu(m,k).
  //  Note that different time-split approaches would give somewhat different results
  // -----------------------------------------------------------------------
  /*
    cloudborne aerosol, so the arrays are dimensioned with pcnst_extd = pcnst*2
  in :: doconvproc_extd[pcnst_extd] ! flag for doing convective transport
  in :: ktop                 ! top level index
  in :: kbot                 ! bottom level index
  in :: iconvtype            ! 1=deep, 2=uw shallow
  in :: dt                   ! Model timestep [s]
  in :: dp[nlev]             ! dp [mb]
  in :: dpdry[nlev]          ! dp [mb]
  in :: cldfrac[nlev]        ! cloud fraction [fraction]
  in :: rhoair[nlev]         ! air density at current i [kg/m3]
  in :: zmagl[nlev]          ! height above surface [m]
  in :: dz                   ! layer thickness [m]
  in :: mu[nlev+1]           ! mu at current i (note nlev+1 dimension, see ma_convproc_tend) [mb/s]
  in :: eudp[nlev]           ! eu(i,k)*dp(i,k) at current i [mb/s]
  in :: gath[pcnst_extd,nlev]   ! gathered tracer array [kg/kg]
  in :: temperature[nlev]  ! Temperature [K]
  in :: aqfrac[pcnst_extd]   ! aqueous fraction of constituent in updraft [fraction]
  in :: icwmr[nlev]    ! Convective cloud water from zm scheme [kg/kg]
  in :: rprd[nlev]     ! Convective precipitation formation rate [kg/kg/s]
  out :: dconudt_wetdep[nlev+1][pcnst_extd1] ! d(conu)/dt by wet removal[kg/kg/s]
  out :: dconudt_activa[nlev+1][pcnst_extd] ! d(conu)/dt by activation [kg/kg/s]
  out :: fa_u[nlev]           ! fractional area of in the updraft
  inout :: conu[nlev+1][pcnst_extd]   ! mix ratio in updraft at interfaces [kg/kg]
  inout :: xx_wcldbase ! w at first cloudy layer [m/s]
  inout :: xx_kcldbase ! level of cloud base
  */
  // clang-format on

  // Threshold below which we treat the mass fluxes as zero [mb/s]
  //  BAD_CONSTANT - used for both compute_downdraft_mixing_ratio and
  //  compute_massflux
  const Real mbsth = 1.e-15;

  // cutoff value of cloud fraction to remove zero cloud [fraction]
  //  BAD_CONSTANT - used for both need a reference for this value
  const Real cldfrac_cut = 0.005;

  // Counter for no. of levels having activation
  int kactcnt = 0;

  // Lowest layer with activation (= cloudbase)
  int kactfirst = 1;

  const int pcnst_extd = ConvProc::pcnst_extd;
  for (int i = 0; i < nlev + 1; ++i)
    for (int j = 0; j < pcnst_extd; ++j)
      dconudt_wetdep(i, j) = dconudt_activa(i, j) = 0.0;

  for (int kk = kbot - 1; ktop <= kk; --kk) {

    // cldfrac = conv cloud fractional area.  This could represent anvil cirrus
    // area, and may not useful for aqueous chem and wet removal calculations
    // adjust to remove zero clouds
    //  cldfrac = cldfrac at current icol (with adjustments) [fraction]
    const Real cldfrac_i = mam4::max(cldfrac[kk], cldfrac_cut);

    const int kp1 = kk + 1;

    // Initialized so that it has a value if the following "if" check yeilds
    // false
    fa_u[kk] = 0;

    // mu_p_eudp = updraft massflux at k, without detrainment between kp1,k
    // [mb/s]
    const Real mu_p_eudp = mu[kp1] + eudp[kk];

    if (mu_p_eudp > mbsth) {
      // if (mu_p_eudp <= mbsth) the updraft mass flux is negligible
      // at base and top of current layer,
      // so current layer is a "gap" between two unconnected updrafts,
      // so essentially skip all the updraft calculations for this layer

      // First apply changes from entrainment
      //  f_ent = fraction of updraft massflux that was entrained
      //  across this layer == eudp/mu_p_eudp [fraction]
      const Real f_ent = utils::min_max_bound(0.0, 1.0, eudp[kk] / mu_p_eudp);
      // The indexing started at 2 for Fortran, so 1 for C++
      // NOTE: conu[kp1] was calculated in the call to compute_wetdep_tend
      // in the last trip through the loop.  The loop iterations over kk
      // are not independent!
      for (int icnst = 1; icnst < pcnst_extd; ++icnst) {
        if (doconvproc_extd[icnst]) {
          conu(kk, icnst) =
              (1.0 - f_ent) * conu(kp1, icnst) + f_ent * gath(kk, icnst);
        }
      }

      // estimate updraft velocity (wup)
      // mean updraft vertical velocity at current level updraft [m/s]
      const Real wup_kk = compute_wup(iconvtype, mu[kk], mu[kp1], cldfrac_i,
                                      rhoair[kk], zmagl[kk]);
      // compute lagrangian transport time (dt_u)
      // -------------------------------------------------------------------
      // There is a bug here that dz is not vertically varying but fixed as
      // the thickness of the lowest level calculated previously in the
      // subroutine ma_convproc_tend. keep it here for C++ refacoring but
      // may need to fix later
      // found by Shuaiqi Tang, 2022
      // -------------------------------------------------------------------
      //  lagrangian transport time in the updraft [s]
      const Real dt_u = mam4::min(dz / wup_kk, dt);

      //  Now apply transformation and removal changes

      // aerosol activation - method 2
      auto dconudt_activa_sub =
          Kokkos::subview(dconudt_activa, kk, Kokkos::ALL());
      auto conu_sub = Kokkos::subview(conu, kk, Kokkos::ALL());
      compute_activation_tend(aero_species, f_ent, cldfrac_i, rhoair[kk],
                              mu[kk], mu[kk + 1], dt_u, wup_kk, icwmr[kk],
                              temperature[kk], kk, kactcnt, kactfirst, conu_sub,
                              dconudt_activa_sub, xx_wcldbase, xx_kcldbase);

      // wet removal
      // NOTE: conu is input/output in this function!  So this call will effect
      // the calculation of conu[kp1] in the next iteration through the loop
      // over levels. This loop can NOT be parallelized over kk!
      auto dconudt_wetdep_sub =
          Kokkos::subview(dconudt_wetdep, kk, Kokkos::ALL());
      compute_wetdep_tend(doconvproc_extd, dt, dt_u, dp[kk], cldfrac_i,
                          mu_p_eudp, aqfrac, icwmr[kk], rprd[kk], conu_sub,
                          dconudt_wetdep_sub);

      // compute updraft fractional area; for update fluxes use
      // *** these must obey  dt_u(k)*mu_p_eudp = dpdry(k)*fa_u(k)
      fa_u[kk] = dt_u * (mu_p_eudp / dpdry[kk]);
    } // "(mu_p_eudp > mbsth)"
  }   // "kk = kbot-1; ktop <= kk; --kk"
}
// ======================================================================================
template <typename SubView, typename ConstSubView>
KOKKOS_INLINE_FUNCTION void
ma_convproc_tend(const AeroSpeciesView &aero_species,
                 const Kokkos::View<Real *>
                     scratch1Dviews[ConvProc::Col1DViewInd::NumScratch],
                 const int nlev, const ConvProc::convtype convtype,
                 const Real dt, const Real temperature[/* nlev */],
                 const Real pmid[/* nlev */], ConstSubView qnew,
                 const Real du[/* nlev */], const Real eu[/* nlev */],
                 const Real ed[/* nlev */], const Real dp[/* nlev */],
                 const Real dpdry[/* nlev */], const int ktop, const int kbot,
                 const int mmtoo_prevap_resusp[/* aero_model::pcnst */],
                 const Real cldfrac[/* nlev */], const Real icwmr[/* nlev */],
                 const Real rprd[/* nlev */], const Real evapc[/* nlev */],
                 SubView dqdt, const bool doconvproc[/* aero_model::pcnst */],
                 Real qsrflx[/* aero_model::pcnst */][nsrflx],
                 const int species_class[/* aero_model::pcnst */],
                 Real &xx_mfup_max, Real &xx_wcldbase, int &xx_kcldbase) {
  // clang-format off
  /*
  // ----------------------------------------------------------------------- 
  //  
  //  Purpose: 
  //  Convective transport of trace species.
  //  The trace species need not be conservative, and source/sink terms for
  //     activation, resuspension, aqueous chemistry and gas uptake, and
  //     wet removal are all applied.
  //  Currently this works with the ZM deep convection, but we should be able
  //     to adapt it for both Hack and McCaa shallow convection
  // 
  // 
  //  Compare to subr convproc which does conservative trace species.
  // 
  //  A distinction between "moist" and "dry" mixing ratios is not currently made.
  //  (P. Rasch comment:  Note that we are still assuming that the tracers are 
  //   in a moist mixing ratio this will change soon)
  // 
  //  
  //  Method: 
  //  Computes tracer mixing ratios in updraft and downdraft "cells" in a
  //  Lagrangian manner, with source/sinks applied in the updraft other.
  //  Then computes grid-cell-mean tendencies by considering
  //     updraft and downdraft fluxes across layer boundaries
  //     environment subsidence/lifting fluxes across layer boundaries
  //     sources and sinks in the updraft
  //     resuspension of activated species in the grid-cell as a whole
  // 
  //  Note1:  A better estimate or calculation of either the updraft velocity
  //          or fractional area is needed.
  //  Note2:  If updraft area is a small fraction of over cloud area, 
  //          then aqueous chemistry is underestimated.  These are both
  //          research areas.
  //  
  //  Authors: O. Seland and R. Easter, based on convtran by P. Rasch
  //  
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  // 
  //  Input arguments
  // 
   in :: convtype  ! identifies the type of
                   ! convection ("deep", "shcu")
   in :: gas_pcnst         ! number of tracers to transport
   in :: dt                ! Model timestep [s]
   in :: temperature[nlev]     ! Temperature [K]
   in :: qnew[nlev][aero_model::pcnst]      ! Tracer array including moisture [kg/kg]

   in :: du[nlev]    ! Mass detrain rate from updraft [1/s]
   in :: eu[nlev]    ! Mass entrain rate into updraft [1/s]
   in :: ed[nlev]    ! Mass entrain rate into downdraft [1/s]
      *** note1 - mu, md, eu, ed, du, dp, dpdry are GATHERED ARRAYS ***
      *** note2 - mu and md units are (mb/s), which is used in the zm_conv code
                - eventually these should be changed to (kg/m2/s)
      *** note3 - eu, ed, du are "d(massflux)/dp" (with dp units = mb), and are all >= 0

   in :: dp[nlev]    ! Delta pressure between interfaces [mb]
   in :: dpdry[nlev] ! Delta dry-pressure [mb]

                     ! Cloud-flux top    
                     ! Layers between kbot,ktop have mass fluxes
                     !    but not all have cloud water, because the
                     !    updraft starts below the cloud base
   in :: ktop        ! Index of cloud top for column
   in :: kbot         Cloud-flux bottom layer for current i (=mx(i))

   in :: cldfrac[nlev]  ! Convective cloud fractional area [fraction]
   in :: icwmr[nlev]    ! Convective cloud water from zhang [kg/kg]
   in :: rprd[nlev]     ! Convective precipitation formation rate [kg/kg/s]
   in :: evapc[nlev]    ! Convective precipitation evaporation rate [kg/kg/s]

   out:: dqdt[nlev][aero_model::pcnst]  ! Tracer tendency array [kg/kg/s]
   in :: doconvproc[aero_model::pcnst] ! flag for doing convective transport
   out:: qsrflx[pcnst][nsrflx]
         ! process-specific column tracer tendencies [kg/m2/s]
         !  1 = activation   of interstial to conv-cloudborne
         !  2 = resuspension of conv-cloudborne to interstital
         !  3 = aqueous chemistry (not implemented yet, so zero)
         !  4 = wet removal
         !  5 = actual precip-evap resuspension (what actually is applied to a species)
         !  6 = pseudo precip-evap resuspension (for history file) 
   in :: species_class[:]  ! specify what kind of species it is. defined at physconst.F90
                                                ! undefined  = 0
                                                ! cldphysics = 1
                                                ! aerosol    = 2
                                                ! gas        = 3
                                                ! other      = 4
   out :: xx_mfup_max   ! diagnostic field of column maximum updraft mass flux [mb/s]
   out :: xx_wcldbase
   out :: xx_kcldbase

  */
  // clang-format on
  const int pcnst_extd = ConvProc::pcnst_extd;
  const int pcnst = aero_model::pcnst;
  EKAT_KERNEL_REQUIRE(convtype == ConvProc::Deep || convtype == ConvProc::Uwsh);
  // iconvtype       ! 1=deep, 2=uw shallow
  const int iconvtype = convtype == ConvProc::Deep ? 1 : 2;
  // iflux_method    ! 1=as in convtran (deep), 2=simpler
  const int iflux_method = convtype == ConvProc::Deep ? 1 : 2;

  const Real hund_ovr_g = 100.0 / Constants::gravity;
  const Real rair = Constants::r_gas_dry_air;

  //  md, mu, are all "dry" mass fluxes
  //  mu(1+nlev)    ! Updraft mass flux (positive) [mb/s]
  //  md(1+nlev)    ! Downdraft mass flux (negative) [mb/s]
  Kokkos::View<Real *> mu = scratch1Dviews[ConvProc::Col1DViewInd::mu];
  Kokkos::View<Real *> md = scratch1Dviews[ConvProc::Col1DViewInd::md];
  // eddp(nlev)           ! ed(k)*dp(k) at current i [mb/s]
  // eudp(nlev)           ! eu(k)*dp(k) at current i [mb/s]
  // dddp(nlev)           ! dd(k)*dp(k) at current i [mb/s]
  // dudp(nlev)           ! du(k)*dp(k) at current i [mb/s]
  Kokkos::View<Real *> eddp = scratch1Dviews[ConvProc::Col1DViewInd::eddp];
  Kokkos::View<Real *> eudp = scratch1Dviews[ConvProc::Col1DViewInd::eudp];
  Kokkos::View<Real *> dddp = scratch1Dviews[ConvProc::Col1DViewInd::dddp];
  Kokkos::View<Real *> dudp = scratch1Dviews[ConvProc::Col1DViewInd::dudp];
  //  rhoair(pver)       ! air density at [kg/m3]
  Kokkos::View<Real *> rhoair = scratch1Dviews[ConvProc::Col1DViewInd::rhoair];
  // zmagl(nlev)         ! working height above surface [m]
  Kokkos::View<Real *> zmagl = scratch1Dviews[ConvProc::Col1DViewInd::zmagl];

  //  gath(nlev,pcnst_extd)   ! gathered tracer array [kg/kg]
  //  chat(nlev+1,pcnst_extd)   ! mix ratio in env at interfaces  [kg/kg]
  //  cond(nlev+1,pcnst_extd)   ! mix ratio in downdraft at interfaces [kg/kg]
  //  conu(nlev+1,pcnst_extd)   ! mix ratio in updraft at interfaces [kg/kg]
  Kokkos_2D_View gath =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::gath].data(), nlev,
          pcnst_extd);
  Kokkos_2D_View chat =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::chat].data(), nlev + 1,
          pcnst_extd);
  Kokkos_2D_View cond =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::cond].data(), nlev + 1,
          pcnst_extd);
  Kokkos_2D_View conu =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::conu].data(), nlev + 1,
          pcnst_extd);

  // dconudt_activa(nlev+1,pcnst_extd) ! d(conu)/dt by activation [kg/kg/s]
  // dconudt_wetdep(nlev+1,pcnst_extd) ! d(conu)/dt by wet removal [kg/kg/s]
  Kokkos_2D_View dconudt_activa =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::dconudt_activa].data(),
          nlev + 1, pcnst_extd);
  Kokkos_2D_View dconudt_wetdep =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::dconudt_wetdep].data(),
          nlev + 1, pcnst_extd);

  // fa_u(nlev)           ! fractional area of in the updraft [fraction]
  Kokkos::View<Real *> fa_u = scratch1Dviews[ConvProc::Col1DViewInd::fa_u];

  // dcondt(nlev,pcnst_extd)  ! grid-average TMR tendency for current column
  // [kg/kg/s]
  Kokkos_2D_View dcondt =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::dcondt].data(), nlev,
          pcnst_extd);

  // dcondt_wetdep(nlev,pcnst_extd) ! portion of dcondt from wet deposition
  // [kg/kg/s] dcondt_prevap(nlev, pcnst_extd) ! portion of dcondt from precip
  // evaporation [kg/kg/s] dcondt_prevap_hist(nlev, pcnst_extd) ! similar but
  // used for history output [kg/kg/s] dcondt_resusp(nlev, pcnst_extd) ! portion
  // of dcondt from resuspension [kg/kg/s]
  Kokkos_2D_View dcondt_wetdep =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::dcondt_wetdep].data(), nlev,
          pcnst_extd);
  Kokkos_2D_View dcondt_prevap =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::dcondt_prevap].data(), nlev,
          pcnst_extd);
  Kokkos_2D_View dcondt_prevap_hist =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::dcondt_prevap_hist].data(),
          nlev, pcnst_extd);
  Kokkos_2D_View dcondt_resusp =
      Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
          scratch1Dviews[ConvProc::Col1DViewInd::dcondt_resusp].data(), nlev,
          pcnst_extd);

  // sumactiva(pcnst_extd)    ! sum (over layers) of dp*dconudt_activa [kg/kg/s]
  // sumaqchem(pcnst_extd)    ! sum (over layers) of dp*dconudt_aqchem [kg/kg/s]
  // sumprevap(pcnst_extd)    ! sum (over layers) of dp*dcondt_prevap [kg/kg/s]
  // sumprevap_hist(pcnst_extd) ! sum (over layers) of dp*dcondt_prevap_hist
  // [kg/kg/s] sumresusp(pcnst_extd)    ! sum (over layers) of dp*dcondt_resusp
  // [kg/kg/s] sumwetdep(pcnst_extd)    ! sum (over layers) of dp*dconudt_wetdep
  // [kg/kg/s]
  Kokkos::View<Real *> sumactiva =
      scratch1Dviews[ConvProc::Col1DViewInd::sumactiva];
  Kokkos::View<Real *> sumaqchem =
      scratch1Dviews[ConvProc::Col1DViewInd::sumaqchem];
  Kokkos::View<Real *> sumprevap =
      scratch1Dviews[ConvProc::Col1DViewInd::sumprevap];
  Kokkos::View<Real *> sumprevap_hist =
      scratch1Dviews[ConvProc::Col1DViewInd::sumprevap_hist];
  Kokkos::View<Real *> sumresusp =
      scratch1Dviews[ConvProc::Col1DViewInd::sumresusp];
  Kokkos::View<Real *> sumwetdep =
      scratch1Dviews[ConvProc::Col1DViewInd::sumwetdep];

  //  q(nlev,pcnst)      ! q(k,m) at current i [kg/kg]
  auto q = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
      scratch1Dviews[ConvProc::Col1DViewInd::q].data(), nlev, pcnst);
  for (int i = 0; i < nlev; ++i)
    for (int j = 0; j < pcnst; ++j)
      q(i, j) = qnew(i, j);

  // precip-borne aerosol
  //    dcondt_wetdep is kgaero/kgair/s
  //    wd_flux = tmpdp*dcondt_wetdep is mb*kgaero/kgair/s
  //    dcondt_prevap = del_wd_flux_evap/dpdry is kgaero/kgair/s
  // so this works ok too
  //
  // *** dilip switched from tmpdg (or dpdry) to tmpdpg = tmpdp/gravit
  // that is incorrect, but probably does not matter
  //    for aerosol, wd_flux units do not matter
  //        only important thing is that tmpdp (or tmpdpg) is used
  //        consistently when going from dcondt to wd_flux then to dcondt
  Kokkos::View<Real *> wd_flux =
      scratch1Dviews[ConvProc::Col1DViewInd::wd_flux];

  // initiate output variables
  for (int i = 0; i < pcnst; ++i)
    for (int j = 0; j < nsrflx; ++j)
      qsrflx[i][j] = 0;
  for (int i = 0; i < nlev; ++i)
    for (int j = 0; j < pcnst; ++j)
      dqdt(i, j) = 0;
  xx_mfup_max = 0;
  xx_wcldbase = 0;
  xx_kcldbase = 0;
  // flag for doing convective transport
  bool doconvproc_extd[pcnst_extd] = {};
  // aqueous fraction of constituent in updraft [fraction]
  Real aqfrac[pcnst_extd] = {};
  //  set doconvproc_extd (extended array) values
  //  inititialize aqfrac to 1.0 for activated aerosol species, 0.0 otherwise
  set_cloudborne_vars(doconvproc, aqfrac, doconvproc_extd);

  // Load some variables in current column for further subroutine use
  for (int kk = 0; kk < nlev; ++kk)
    rhoair[kk] = pmid[kk] / (rair * temperature[kk]);

  //  load tracer mixing ratio array, which will be updated at the end of each
  //  jtsub interation

  // calculate dry mass fluxes at cloud layer
  compute_massflux(nlev, ktop, kbot, dpdry, du, eu, ed, mu.data(), md.data(),
                   xx_mfup_max);

  //  compute entraintment*dp and detraintment*dp and calculate ntsub
  int ntsub = 0;
  compute_ent_det_dp(nlev, ktop, kbot, dt, dpdry, mu.data(), md.data(), du, eu,
                     ed, ntsub, eudp.data(), dudp.data(), eddp.data(),
                     dddp.data());

  // calculate height of layer interface above ground
  compute_midlev_height(nlev, dpdry, rhoair.data(), zmagl.data());

  for (int jtsub = 0; jtsub < ntsub; ++jtsub) {

    // initialize some tracer mixing ratio arrays
    initialize_tmr_array(nlev, iconvtype, doconvproc_extd, q, gath, chat, conu,
                         cond);

    // Compute updraft mixing ratios from cloudbase to cloudtop
    // ---------------------------------------------------------------------------
    // there is a bug here for dz, which is not a vertically-varying profile.
    // see the subroutine compute_updraft_mixing_ratio for the bug information
    // by Shuaiqi Tang, 2022
    // ---------------------------------------------------------------------------
    const Real dz = dpdry[0] * hund_ovr_g / rhoair[0];

    compute_updraft_mixing_ratio(
        aero_species, doconvproc_extd, nlev, ktop, kbot, iconvtype, dt, dp,
        dpdry, cldfrac, rhoair.data(), zmagl.data(), dz, mu.data(), eudp.data(),
        gath, temperature, aqfrac, icwmr, rprd, fa_u.data(), dconudt_wetdep,
        dconudt_activa, conu, xx_wcldbase, xx_kcldbase);

    // Compute downdraft mixing ratios from cloudtop to cloudbase
    compute_downdraft_mixing_ratio(doconvproc_extd, ktop, kbot, md.data(),
                                   eddp.data(), gath, cond);

    // Now compute fluxes and tendencies
    // NOTE:  The approach used in convtran applies to inert tracers and
    //        must be modified to include source and sink terms
    initialize_dcondt(doconvproc_extd, iflux_method, ktop, kbot, nlev, dpdry,
                      fa_u.data(), mu.data(), md.data(), chat, gath, conu, cond,
                      dconudt_activa, dconudt_wetdep, dudp.data(), dddp.data(),
                      eudp.data(), eddp.data(), dcondt);

    // compute dcondt_wetdep for next subroutine
    for (int i = 0; i < nlev; ++i)
      for (int j = 0; j < pcnst_extd; ++j)
        dcondt_wetdep(i, j) = 0.0;
    for (int kk = ktop; kk < kbot; ++kk) {
      // simply cancelling dpdry causes BFB test fail
      const Real fa_u_dp = fa_u[kk] * dpdry[kk];
      // The indexing started at 2 for Fortran, so 1 for C++
      for (int icnst = 1; icnst < pcnst_extd; ++icnst) {
        if (doconvproc_extd[icnst]) {
          dcondt_wetdep(kk, icnst) =
              fa_u_dp * dconudt_wetdep(kk, icnst) / dpdry[kk];
        }
      }
    }

    // calculate effects of precipitation evaporation
    ma_precpevap_convproc(ktop, nlev, dcondt_wetdep, rprd, evapc, dpdry,
                          doconvproc_extd, species_class, mmtoo_prevap_resusp,
                          wd_flux, dcondt_prevap, dcondt_prevap_hist, dcondt);

    //  make adjustments to dcondt for activated & unactivated aerosol species
    //     pairs to account any (or total) resuspension of convective-cloudborne
    //     aerosol

    //  usually the updraft ( & downdraft) start ( & end ) at kbot=nlev, but
    //  sometimes kbot < nlev transport, activation, resuspension, and wet
    //  removal only occur between kbot >= k >= ktop resuspension from
    //  evaporating precip can occur at k > kbot when kbot < nlev in the first
    //  version of this routine, the precp evap resusp tendencies for k > kbot
    //  were ignored,
    //     but that is now fixed
    //  this was a minor bug with quite minor affects on the aerosol,
    //     because convective precip evap is (or used to be) much less than
    //     stratiform precip evap )
    //       kbot_prevap = kbot
    //  apply this minor fix when doing resuspend to coarse mode
    const int kbot_prevap = nlev;
    for (int kk = ktop; kk < kbot_prevap; ++kk)
      ma_resuspend_convproc(Kokkos::subview(dcondt, kk, Kokkos::ALL()),
                            Kokkos::subview(dcondt_resusp, kk, Kokkos::ALL()));

    // calculate new column-tendency variables
    compute_column_tendency(
        doconvproc_extd, ktop, kbot_prevap, dpdry, dcondt_resusp, dcondt_prevap,
        dcondt_prevap_hist, dconudt_activa, dconudt_wetdep, fa_u.data(),
        sumactiva.data(), sumaqchem.data(), sumwetdep.data(), sumresusp.data(),
        sumprevap.data(), sumprevap_hist.data());

    // NOTE: update_tendency_final Fortran =
    //   update_tendency_diagnostics C++ + update_tendency_final C++
    //   because of the two loops in this function it was easier
    //   to split the function up so that one of these could
    //   be done in a thread team.
    update_tendency_diagnostics(ntsub, pcnst, doconvproc, sumactiva.data(),
                                sumaqchem.data(), sumwetdep.data(),
                                sumresusp.data(), sumprevap.data(),
                                sumprevap_hist.data(), qsrflx);
    // update tendencies
    for (int kk = ktop; kk < kbot_prevap; ++kk) {
      update_tendency_final(
          ntsub, jtsub, pcnst, dt, Kokkos::subview(dcondt, kk, Kokkos::ALL()),
          doconvproc, Kokkos::subview(dqdt, kk, Kokkos::ALL()),
          Kokkos::subview(q, kk, Kokkos::ALL()));
    }
  } // of the main "for jtsub = 0, ntsub" loop
}

// =========================================================================================
template <typename SubView, typename ConstSubView>
KOKKOS_INLINE_FUNCTION void
ma_convproc_dp_intr(const AeroSpeciesView &aero_species,
                    const Kokkos::View<Real *>
                        scratch1Dviews[ConvProc::Col1DViewInd::NumScratch],
                    const int nlev, const Real temperature[/* nlev */],
                    const Real pmid[/* nlev */], const Real dpdry[/* nlev */],
                    const Real dt, const Real cldfrac[/* nlev */],
                    const Real icwmr[/* nlev */], const Real rprddp[/* nlev */],
                    const Real evapcdp[/* nlev */], const Real du[/* nlev */],
                    const Real eu[/* nlev */], const Real ed[/* nlev */],
                    const Real dp[/* nlev */], const int ktop, const int kbot,
                    ConstSubView qnew,
                    const int species_class[/* aero_model::pcnst */],
                    const int mmtoo_prevap_resusp[/* aero_model::pcnst */],
                    SubView dqdt, Real qsrflx[/* aero_model::pcnst */][nsrflx],
                    bool dotend[aero_model::pcnst]) {

  // clang-format off
  // ----------------------------------------------------------------------- 
  //  
  //  Purpose: 
  //  Convective cloud processing (transport, activation/resuspension,
  //     wet removal) of aerosols and trace gases.
  //     (Currently no aqueous chemistry and no trace-gas wet removal)
  //  Does aerosols    when convproc_do_aer is .true.
  //  Does trace gases when convproc_do_gas is .true.
  // 
  //  This routine does deep convection
  //  Uses mass fluxes, cloud water, precip production from the
  //     convective cloud routines
  //  
  //  Author: R. Easter
  //  
  // -----------------------------------------------------------------------

  /* 
! Arguments
   in    :: dpdry[nlev]           ! layer delta-p-dry [mb]
   in    :: temperature[nlev]         ! Temperature [K]
   in    :: pmid[nlev]      ! Pressure at model levels [Pa]

   in    :: dt                   ! delta t (model time increment) [s]
   in    :: qnew[nlev][pcnst]     ! tracer mixing ratio including water vapor [kg/kg]
   in    :: nsrflx               ! last dimension of qsrflx

   in    :: cldfrac[nlev] ! Deep conv cloud fraction [0-1]
   in    :: icwmr[nlev] ! Deep conv cloud condensate (in cloud) [kg/kg]
   in    :: rprddp[nlev]  ! Deep conv precip production (grid avg) [kg/kg/s]
   in    :: evapcdp[nlev] ! Deep conv precip evaporation (grid avg) [kg/kg/s]

   in    :: du[nlev]   ! Mass detrain rate from updraft [1/s]
   in    :: eu[nlev]   ! Mass entrain rate into updraft [1/s]
   in    :: ed[nlev]   ! Mass entrain rate into downdraft [1/s]
         ! eu, ed, du are "d(massflux)/dp" and are all positive
   in    :: dp[nlev]   ! Delta pressure between interfaces [mb]

   in    :: ktop              ! Index of cloud top
   in    :: kbot              ! Index of cloud bottom
   in    :: species_class[:]  ! species index
   in    :: mmtoo_prevap_resusp values are:
             >=0 for aerosol mass species with    coarse mode counterpart
             -2 for aerosol mass species WITHOUT coarse mode counterpart
             -3 for aerosol number species
             -1 for other species

   out   :: dotend[pcnst]        ! if do tendency
   inout :: dqdt[nlev][pcnst]     ! time tendency of q [kg/kg/s]
   inout :: qsrflx[pcnst,nsrflx] ! process-specific column tracer tendencies (see ma_convproc_intr for more information) [kg/m2/s]

  */
  // clang-format on
  // Local variables

  // diagnostics variables to write out.
  // xx_kcldbase is filled with index kk. maybe better define as integer
  // keep it in C++ refactoring for BFB comparison (by Shuaiqi Tang, 2022)
  Real xx_mfup_max, xx_wcldbase;
  int xx_kcldbase;

  // turn on/off calculations for aerosols and trace gases
  const bool convproc_do_aer = true;
  const bool convproc_do_gas = false;
  assign_dotend(species_class, convproc_do_aer, convproc_do_gas, dotend);

  // clang-format off
  //
  // do ma_convproc_tend call
  //
  // question/issue - when computing first-order removal rate for convective cloud water,
  //    should dlf be included as is done in wetdepa?
  // detrainment does not change the in-cloud (= in updraft) cldwtr mixing ratio
  // when you have detrainment, the updraft air mass flux is decreasing with height,
  //    and the cldwtr flux may be decreasing also, 
  //    but the in-cloud cldwtr mixing ratio is not changed by detrainment itself
  // this suggests that wetdepa is incorrect, and dlf should not be included
  //
  // if dlf should be included, then you want to calculate
  //    rprddp / (cldfrac*icwmr + dt*(rprddp + dlfdp)]
  // so need to pass both rprddp and dlfdp to ma_convproc_tend
  //
  // clang-format on

  ma_convproc_tend(aero_species, scratch1Dviews, nlev, ConvProc::Deep, dt,
                   temperature, pmid, qnew, du, eu, ed, dp, dpdry, ktop, kbot,
                   mmtoo_prevap_resusp, cldfrac, icwmr, rprddp, evapcdp, dqdt,
                   dotend, qsrflx, species_class, xx_mfup_max, xx_wcldbase,
                   xx_kcldbase);
}

// =========================================================================================
KOKKOS_INLINE_FUNCTION
void ma_convproc_intr(
    const ThreadTeam &team, const AeroSpeciesView &aero_species,
    const Kokkos::View<Real *>
        scratch1Dviews[ConvProc::Col1DViewInd::NumScratch],
    const bool convproc_do_aer, const bool convproc_do_gas, const int nlev,
    const Real temperature[/* nlev */], const Real pmid[/* nlev */],
    const Real dpdry[/* nlev */], const Real dt,
    const Real dp_frac[/* nlev */], const Real icwmrdp[/* nlev */],
    const Real rprddp[/* nlev */], const Real evapcdp[/* nlev */],
    const Real dlf[/* nlev */],
    const Real du[/* nlev */], const Real eu[/* nlev */],
    const Real ed[/* nlev */],
    const Real dp[/* nlev */], const int ktop, const int kbot,
    const int species_class[aero_model::pcnst],
    const int mmtoo_prevap_resusp[aero_model::pcnst],
    const Diagnostics::ColumnTracerView state_q,
    Diagnostics::ColumnTracerView ptend_q, bool ptend_lq[aero_model::pcnst],
    Real aerdepwetis[aero_model::pcnst]) {

  //-----------------------------------------------------------------------
  //
  // Purpose:
  // Convective cloud processing (transport, activation/resuspension,
  //    wet removal) of aerosols and trace gases.
  //    (Currently no aqueous chemistry and no trace-gas wet removal)
  // Does aerosols    when convproc_do_aer is .true.
  // Does trace gases when convproc_do_gas is .true.
  //
  // Does deep convection
  // Uses mass fluxes, cloud water, precip production from the
  //    convective cloud routines
  //
  // Author: R. Easter
  //
  //-----------------------------------------------------------------------

  // clang-format off
  // Arguments
  /*  
  in    :: dpdry[nlev]           ! layer delta-p-dry [mb]
  in    :: temperature[nlev]     ! Temperature [K]
  in    :: pmid[nlev]            ! Pressure at model levels [Pa]

  in    :: dt                        ! 2 delta t [s]
                                ! from input this is just model time step, 
                                ! not sure why it is "2" delta t. Shuaiqi Tang 2022
  in    :: dp_frac[nlev]       ! Deep conv cloud frac [fraction]
  in    :: icwmrdp[nlev]         ! Deep conv cloud condensate (in cloud) [kg/kg]
  in    :: rprddp[nlev]        ! Deep conv precip production (grid avg) [kg/kg/s]
  in    :: evapcdp[nlev]       ! Deep conv precip evaporation (grid avg) [kg/kg/s]
  in    :: dlf[nlev]           ! Tot  conv cldwtr detrainment (grid avg) [kg/kg/s]
  inout :: aerdepwetis[pcnst]  ! aerosol wet deposition (interstitial) [kg/m2/s]
                       ! eu, ed, du are "d(massflux)/dp" and are all positive
  in    :: eu[nlev]    ! Mass entrain rate into updraft [1/s]
  in    :: ed[nlev]    ! Mass entrain rate into downdraft [1/s]
  in    :: du[nlev]    ! Mass detrain rate from updraft [1/s]
  in    :: dp[nlev]    ! Delta pressure between interfaces [mb]
  in    :: ktop           ! Index of cloud top (updraft top) for each column in w grid
  in    :: kbot         ! Index of cloud base (level of maximum moist static energy) for each column in w grid
  in    :: species_class[aero_model::pcnst]  ! species index defined as
          spec_class::undefined  = 0
          spec_class::cldphysics = 1
          spec_class::aerosol    = 2
          spec_class::gas        = 3
          spec_class::other      = 4
  
  in  mmtoo_prevap_resaero_model::pcnst[aero_model::pcnst]
        pointers for resuspension mmtoo_prevap_resusp values are
           >=0 for aerosol mass species with    coarse mode counterpart
           -2 for aerosol mass species WITHOUT coarse mode counterpart
           -3 for aerosol number species
           -1 for other species
  */
  // clang-format on

  auto dqdt = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
      scratch1Dviews[ConvProc::Col1DViewInd::dqdt].data(), nlev,
      aero_model::pcnst);

  auto dlfdp = Kokkos::View<Real *, Kokkos::MemoryUnmanaged>(
      scratch1Dviews[ConvProc::Col1DViewInd::dlfdp].data(), nlev);

  for (int j = 0; j < nlev; ++j)
    for (int i = 0; i < aero_model::pcnst; ++i)
      dqdt(j, i) = ptend_q(j, i);

  // qnew will update in the subroutines but not update back to state%q
  auto qnew = Diagnostics::ColumnTracerView(
      scratch1Dviews[ConvProc::Col1DViewInd::qnew].data(), nlev,
      aero_model::pcnst);
  EKAT_KERNEL_ASSERT(state_q.extent_int(0) == nlev);
  EKAT_KERNEL_ASSERT(state_q.extent_int(1) <= aero_model::pcnst);
  for (int i = 0; i < nlev; ++i)
    for (int j = 0; j < aero_model::pcnst; ++j)
      qnew(i, j) = state_q(i, j);

  // if do tendency
  bool dotend[aero_model::pcnst];
  for (int i = 0; i < aero_model::pcnst; ++i)
    dotend[i] = ptend_lq[i];

  //
  // prepare for processing
  //

  // The following loop can be done in parallel even though ptend_lq is
  // overwritten each time I think it is OK since it is overwritten the same
  // from each thread.
  for (int kk = 0; kk < nlev; ++kk)
    update_qnew_ptend(dotend, false, Kokkos::subview(dqdt, kk, Kokkos::ALL()),
                      dt, ptend_lq, Kokkos::subview(ptend_q, kk, Kokkos::ALL()),
                      Kokkos::subview(qnew, kk, Kokkos::ALL()));

  if (convproc_do_aer || convproc_do_gas) {
    //
    // do deep conv processing
    //
    Real qsrflx[aero_model::pcnst][nsrflx] = {};
    for (int j = 0; j < nlev; ++j)
      for (int i = 0; i < aero_model::pcnst; ++i)
        dqdt(j, i) = 0;
    for (int j = 0; j < nlev; ++j)
      dlfdp[j] = mam4::max((dlf[j]), 0.0);
    ma_convproc_dp_intr(aero_species, scratch1Dviews, nlev, temperature, pmid,
                        dpdry, dt, dp_frac, icwmrdp, rprddp, evapcdp, du, eu,
                        ed, dp, ktop, kbot, qnew, species_class,
                        mmtoo_prevap_resusp, dqdt, qsrflx, dotend);
    // apply deep conv processing tendency
    for (int kk = 0; kk < nlev; ++kk)
      update_qnew_ptend(dotend, true, Kokkos::subview(dqdt, kk, Kokkos::ALL()),
                        dt, ptend_lq,
                        Kokkos::subview(ptend_q, kk, Kokkos::ALL()),
                        Kokkos::subview(qnew, kk, Kokkos::ALL()));
    // update variables for output
    for (int icnst = 0; icnst < aero_model::pcnst; ++icnst) {
      // this used for surface coupling:
      //  4 = wet removal
      //  5 = actual precip-evap resuspension (what actually is applied to a
      //  species)
      if (dotend[icnst] &&
          species_class[icnst] == ConvProc::species_class::aerosol)
        aerdepwetis[icnst] += qsrflx[icnst][4] + qsrflx[icnst][5];
    }
  } // (convproc_do_aer || convproc_do_gas)
}
} // namespace convproc

// TODO: The ThreadTeam passed to compute_tendencies is not currently used
// as there are some low-level function that can not be parallel over the
// column due to the integration like algorithms.  But there are some
// functions that could be parallel over the column and these need to
// be teased out and run with the team.
KOKKOS_INLINE_FUNCTION
void ConvProc::compute_tendencies(const AeroConfig &config,
                                  const ThreadTeam &team, Real t, Real dt,
                                  const Atmosphere &atmosphere,
                                  const Prognostics &prognostics,
                                  const Diagnostics &diagnostics,
                                  const Tendencies &tendencies) const {

  const bool convproc_do_aer = config_.convproc_do_aer;
  const bool convproc_do_gas = config_.convproc_do_gas;
  const int ktop = config_.ktop;
  const int kbot = config_.kbot;
  int species_class[aero_model::pcnst];
  for (int i = 0; i < aero_model::pcnst; ++i)
    species_class[i] = config_.species_class[i];
  int mmtoo_prevap_resusp[aero_model::pcnst];
  ;
  for (int i = 0; i < aero_model::pcnst; ++i)
    mmtoo_prevap_resusp[i] = config_.mmtoo_prevap_resusp[i];

  const int nlev = atmosphere.num_levels();
  const Real *temperature = atmosphere.temperature.data();
  const Real *pmid = atmosphere.pressure.data();
  // dpdry =  Delta dry-pressure [mb]
  const Real *dpdry = diagnostics.hydrostatic_dry_dp.data();
  // dp_frac = Deep conv cloud frac [fraction]
  const Real *dp_frac = diagnostics.deep_convective_cloud_fraction.data();
  // Deep cloud convective condensate [kg/kg]
  const Real *icwmrdp = diagnostics.deep_convective_cloud_condensate.data();
  // Deep convective precipitation production (grid avg) [kg/kg/s]
  const Real *rprddp =
      diagnostics.deep_convective_precipitation_production.data();
  // Deep convective precipitation evaporation (grid avg) [kg/kg/s]
  const Real *evapcdp =
      diagnostics.deep_convective_precipitation_evaporation.data();
  // Shallow+Deep convective detrainment [kg/kg/s]
  const Real *dlftot = diagnostics.total_convective_detrainment.data();

  // Next three are "d(massflux)/dp" and are all positive [1/s]
  const Real *eu = diagnostics.mass_entrain_rate_into_updraft.data();
  const Real *ed = diagnostics.mass_entrain_rate_into_downdraft.data();
  const Real *du = diagnostics.mass_detrain_rate_from_updraft.data();

  // Delta pressure between interfaces [mb]
  const Real *dp = diagnostics.delta_pressure.data();

  // Tracer mixing ratio (TMR) including water vapor [kg/kg]
  const auto state_q = diagnostics.tracer_mixing_ratio;
  // Time tendency of tracer mixing ratio (TMR) [kg/kg/s]
  auto ptend_q = diagnostics.d_tracer_mixing_ratio_dt;

  // Diagnostic values that might be of use but not sure.
  // ptend_lq is just a flag that signifies if the gas was updated.
  bool ptend_lq[aero_model::pcnst] = {};
  // Aerosol wet deposition (interstitial) [kg/m2/s]
  Real aerdepwetis[aero_model::pcnst] = {};
  convproc::ma_convproc_intr(
      team, config.aero_species, scratch1Dviews, convproc_do_aer,
      convproc_do_gas, nlev, temperature, pmid, dpdry, dt, dp_frac,
      icwmrdp, rprddp, evapcdp, dlftot,
      du, eu, ed, dp, ktop, kbot, species_class,
      mmtoo_prevap_resusp, state_q, ptend_q, ptend_lq, aerdepwetis);
}
} // namespace mam4
#endif
