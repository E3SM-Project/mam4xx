#ifndef MAM4XX_MO_SETSOX_HPP
#define MAM4XX_MO_SETSOX_HPP

#include "aero_config.hpp"
#include "aero_modes.hpp"
#include "mam4_math.hpp"
#include "mam4_types.hpp"

namespace mam4 {

namespace mo_setsox {

// mo_setsox-specific configuration
struct Config {

  // number of vertical levels
  int pver = mam4::nlev;
  int modeptr_accum = int(ModeIndex::Accumulation);
  /*
  these are the entries/indices in the gas-phase species array
  from: eam/src/chemistry/pp_linoz_mam4_resus_mom_soag/mo_sim_dat.F90#L23)
  solsym(: 31) = (/ 'O3', 'H2O2', 'H2SO4', 'SO2', 'DMS',  'SOAG', 'so4_a1',
                    'pom_a1', 'soa_a1', 'bc_a1',  'dst_a1', 'ncl_a1',
                    'mom_a1', 'num_a1', 'so4_a2',  'soa_a2', 'ncl_a2',
                    'mom_a2', 'm_a2', 'dst_a3',  'ncl_a3', 'so4_a3', 'bc_a3',
                    'pom_a3', 'soa_a3', 'mom_a3', 'num_a3', 'pom_a4',
                    'bc_a4', 'mom_a4', 'num_a4' /)
  Fortran-indexed:
  1: 'O3', 2: 'H2O2', 3: 'H2SO4', 4: 'SO2', 5: 'DMS', 6: 'SOAG', 7: 'so4_a1',
  8: 'pom_a1', 9: 'soa_a1', 10: 'bc_a1', 11: 'dst_a1', 12: 'ncl_a1',
  13: 'mom_a1', 14: 'num_a1', 15: 'so4_a2', 16: 'soa_a2', 17: 'ncl_a2',
  18: 'mom_a2', 19: 'num_a2', 20: 'dst_a3', 21: 'ncl_a3', 22: 'so4_a3',
  23: 'bc_a3', 24: 'pom_a3', 25: 'soa_a3', 26: 'mom_a3', 27: 'num_a3',
  28: 'pom_a4', 29: 'bc_a4', 30: 'mom_a4', 31: 'num_a4'
  c++ indexed:
  0: 'O3', 1: 'H2O2', 2: 'H2SO4', 3: 'SO2', 4: 'DMS', 5: 'SOAG', 6: 'so4_a1',
  7: 'pom_a1', 8: 'soa_a1', 9: 'bc_a1', 10: 'dst_a1', 11: 'ncl_a1',
  12: 'mom_a1', 13: 'num_a1', 14: 'so4_a2', 15: 'soa_a2', 16: 'ncl_a2',
  17: 'mom_a2', 18: 'num_a2', 19: 'dst_a3', 20: 'ncl_a3', 21: 'so4_a3',
  22: 'bc_a3', 23: 'pom_a3', 24: 'soa_a3', 25: 'mom_a3', 26: 'num_a3',
  27: 'pom_a4', 28: 'bc_a4', 29: 'mom_a4', 30: 'num_a4'
  */
  int id_so2 = 3;
  int id_h2o2 = 1;
  int id_o3 = 0;
  int id_h2so4 = 2;

  // ===========================================================================
  //     BAD CONSTANTS (let's just put the global ones in one place for now)
  // ===========================================================================
  // Real lwc = 0.2302286341e-4;
  int lptr_so4_cw_amode[4] = {15, 23, 30, -1};
  // int loffset = 9;
  Real small_value_lwc = 1.0e-8;
  // Real small_value_cf = 1.0e-5;
  Real p0 = 101300.0;
  bool cloud_borne = true;
  bool modal_aerosols = true;
  // universal gas constant (sort of)
  // FIXME: TERRIBLE CONSTANT
  Real Ra = 8314.0 / 101325.0;
  // water acidity
  Real xkw = 1.0e-14;
  // FIXME: AWFUL CONSTANT
  // [cm3/L / Avogadro constant]
  Real const0 = 1.0e3 / 6.023e23;
  // 330 ppm = 330.0e-6 atm
  Real co2g = 330.0e-6;
  int itermax = 20;
  // Real small_value_20 = 1.0e-20;
  // Real small_value_30 = 1.0e-30;
  int numptrcw_amode[AeroConfig::num_modes()] = {22, 27, 35, 39};

  Config() = default;
  Config(const Config &) = default;
  ~Config() = default;
  Config &operator=(const Config &) = default;
};

// struct for the sox_cldaero_create_obj()
// NOTE: maybe ditch this in the future to make life simpler?
struct Cloudconc {
  Real so4c;
  Real xlwc;
  Real so4_fact;
};

KOKKOS_INLINE_FUNCTION
Cloudconc sox_cldaero_create_obj(
    const Real cldfrc, const Real qcw[AeroConfig::num_gas_phase_species()],
    const Real lwc, const Real cfact, const int loffset, const Config config_) {
  /*
  input variables
  cldfrc: cloud fraction [fraction]
  qcw: cloud-borne aerosol [vmr]
  lwc: cloud liquid water content [kg/kg]
  cfact: total atms density total atms density [kg/L]
  loffset: # of tracers in the host model that are not part of MAM
  */

  Real so4c = 0.0;
  Real xlwc = 0.0;

  // xlwc is in-cloud LWC with the unit of [kg/L]
  if (cldfrc > 0.0) {
    // cloud water L(water)/L(air)
    xlwc = lwc * cfact;
    // liquid water in the cloudy fraction of cell
    xlwc = xlwc / cldfrc;
  } else {
    xlwc = 0.0;
  }

  // BAD ASSUMPTION?
  // lptr_so4_cw_amode is a modal variable, but here it's hard-coded to only
  // look at the first 3. Granted, entry 4 is zero, but will it always be?
  int id_so4_1a = config_.lptr_so4_cw_amode[0] - loffset;
  int id_so4_2a = config_.lptr_so4_cw_amode[1] - loffset;
  int id_so4_3a = config_.lptr_so4_cw_amode[2] - loffset;
  so4c = qcw[id_so4_1a] + qcw[id_so4_2a] + qcw[id_so4_3a];

  // with 3-mode, assume so4 is nh4hso4, and so half-neutralized
  // FIXME: BAD CONSTANT
  Real so4_fact = 1.0;
  Cloudconc cldconc;
  cldconc.so4c = so4c;
  cldconc.xlwc = xlwc;
  cldconc.so4_fact = so4_fact;
  return cldconc;
} // end function sox_cldaero_create_obj

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_so2(const Real t_factor, Real &xk, Real &xe, Real &x2) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk, xe and x2 for SO2
  //-----------------------------------------------------------------

  // FIXME: BAD CONSTANTS
  // for so2
  xk = 1.230 * mam4::exp(3120.0 * t_factor);
  xe = 1.7e-2 * mam4::exp(2090.0 * t_factor);
  x2 = 6.0e-8 * mam4::exp(1120.0 * t_factor);
} // end henry_factor_so2

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_co2(const Real t_factor, Real &xk, Real &xe) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk and xe for CO2
  //-----------------------------------------------------------------

  // FIXME: BAD CONSTANTS
  // for co2
  xk = 3.1e-2 * mam4::exp(2423.0 * t_factor);
  xe = 4.3e-7 * mam4::exp(-913.0 * t_factor);
} // end henry_factor_co2

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_h2o2(const Real t_factor, Real &xk, Real &xe) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk and xe for H2O2
  //-----------------------------------------------------------------

  // FIXME: BAD CONSTANTS
  // for h2o2
  xk = 7.4e4 * mam4::exp(6621.0 * t_factor);
  xe = 2.2e-12 * mam4::exp(-3730.0 * t_factor);

} // end henry_factor_h2o2

//===========================================================================
KOKKOS_INLINE_FUNCTION
void henry_factor_o3(const Real t_factor, Real &xk) {
  //-----------------------------------------------------------------
  // get Henry Law parameters xk and xe for O3
  //-----------------------------------------------------------------

  // FIXME: BAD CONSTANTS
  // for o3
  xk = 1.15e-2 * mam4::exp(2560.0 * t_factor);
} // end henry_factor_o3

//===========================================================================
// FIXME: it appears that ynetpos is never used. if this is the correct
// behavior, then the xph calculation below should maybe be done directly and
// this function be deleted
KOKKOS_INLINE_FUNCTION
void calc_ynetpos(const Real yph, const Real fact1_so2, const Real fact2_so2,
                  const Real fact3_so2, const Real fact4_so2, const Real Eco2,
                  const Real Eh2o, const Real Eso4, const Real so4_fact,
                  // out
                  Real &xph, Real &ynetpos) {
  //-----------------------------------------------------------------
  // calculate net positive ions (ynetpos) for iterations in calc_ph_values
  // also calculate H+ concentration (xph) from ph value
  //-----------------------------------------------------------------

  // calc current [H+] from pH
  xph = mam4::pow(10.0, -yph);

  //-----------------------------------------------------------------
  //          ... so2
  //-----------------------------------------------------------------
  Real Eso2 =
      fact1_so2 /
      (1.0 + fact2_so2 * (1.0 + (fact3_so2 / xph) * (1.0 + fact4_so2 / xph)));

  Real tmp_hso3 = Eso2 / xph;
  Real tmp_so3 = tmp_hso3 * 2.0 * fact4_so2 / xph;
  Real tmp_hco3 = Eco2 / xph;
  Real tmp_oh = Eh2o / xph;
  Real tmp_so4 = so4_fact * Eso4;

  // positive ions are H+ only
  Real tmp_pos = xph;
  // all negative ions
  Real tmp_neg = tmp_oh + tmp_hco3 + tmp_hso3 + tmp_so3 + tmp_so4;

  ynetpos = tmp_pos - tmp_neg;
} // end calc_ynetpos

//===========================================================================
KOKKOS_INLINE_FUNCTION
void calc_ph_values(const Real temperature, const Real patm, const Real xlwc,
                    const Real t_factor, const Real xso2, const Real xso4,
                    const Real xhnm, const Real so4_fact, const Real Ra,
                    const Real xkw, const Real const0, const Real co2g,
                    const int itermax,
                    // out
                    bool &converged, Real &xph) {
  /*
  ---------------------------------------------------------------------------
  calculate PH value and H+ concentration

  21-mar-2011 changes by rce
  now uses bisection method to solve the electro-neutrality equation
  3-mode aerosols (where so4 is assumed to be nh4hso4)
        old code set xnh4c = so4c
        new code sets xnh4c = 0, then uses a -1 charge (instead of -2)
       for so4 when solving the electro-neutrality equation
  ---------------------------------------------------------------------------
      implicit none

      real(r8),  intent(in) :: temperature        // temperature [K]
      real(r8),  intent(in) :: patm               // pressure [atm]
      real(r8),  intent(in) :: t_factor           // working variable to
      convert to 25 degC (1/T - 1/[298K]) real(r8),  intent(in) :: xso2 //
      SO2 [mol/mol] real(r8),  intent(in) :: xso4               // SO4
      [mol/mol] real(r8),  intent(in) :: xhnm               // [#/cm3]
      real(r8),  intent(in) :: xlwc               // in-cloud LWC [kg/L]
      real(r8),  intent(in) :: so4_fact           // factor for SO4
      real(r8),  intent(in) :: Ra                 // constant parameter
      real(r8),  intent(in) :: xkw                // constant parameter
      real(r8),  intent(in) :: const0             // constant parameter

      logical,  intent(out) :: converged          // if the method converge
      real(r8), intent(out) :: xph                // H+ ions concentration
      [mol/L]

      // local variables
      integer   :: iter  // iteration number
      real(r8)  :: yph_lo, yph_hi, yph    // pH values, lower and upper
      bounds real(r8)  :: ynetpos_lo, ynetpos_hi // lower and upper bounds of
      ynetpos real(r8)  :: xk, xe, x2     // output parameters in Henry's law
      real(r8)  :: fact1_so2, fact2_so2, fact3_so2, fact4_so2  // SO2 factors
      real(r8)  :: Eh2o, Eco2, Eso4 // effects of species [1/cm3]
      real(r8)  :: ynetpos        // net positive ions

      integer,  parameter :: itermax = 20  // maximum number of iterations
      real(r8), parameter :: co2g = 330.e-6    //330 ppm = 330.e-6 atm
  #include "../yaml/mo_setsox/f90_yaml/calc_ph_values_beg_yml.f90"

  ----------------------------------------
  effect of chemical species
  ----------------------------------------

  -------------- hno3 -------------------
  FORTRAN refactoring: not incorporated in MAM4

  -------------- nh3 -------------------
  FORTRAN refactoring: not incorporated in MAM4

  -------------- so2 -------------------
  previous code
     heso2(i,k)  = xk*(1.0 + xe/xph(i,k)*(1.0 + x2/xph(i,k)))
     px = heso2(i,k) * Ra * tz * xl
     so2g =  xso2(i,k)/(1.0+ px)
     Eso2 = xk*xe*so2g *patm
  equivalent new code
     heso2 = xk + xk*xe/hplus * xk*xe*x2/hplus**2
     so2g = xso2/(1 + px)
          = xso2/(1 + heso2*ra*tz*xl)
          = xso2/(1 + xk*ra*tz*xl*(1 + (xe/hplus)*(1 + x2/hplus))
     eso2 = so2g*xk*xe*patm
           = xk*xe*patm*xso2/(1 + xk*ra*tz*xl*(1 + (xe/hplus)*(1 + x2/hplus))
           = ( fact1_so2    )/(1 + fact2_so2 *(1 + (fact3_so2/hplus)*(1 +
           fact4_so2/hplus)
     [hso3-] + 2*[so3--] = (eso2/hplus)*(1 + 2*x2/hplus)
  */

  // output parameters in Henry's law
  Real xk, xe, x2;
  // SO2 factors
  Real fact1_so2, fact2_so2, fact3_so2, fact4_so2;
  henry_factor_so2(t_factor,
                   // out
                   xk, xe, x2);
  fact1_so2 = xk * xe * patm * xso2;
  fact2_so2 = xk * Ra * temperature * xlwc;
  fact3_so2 = xe;
  fact4_so2 = x2;

  // Eh2o, Eco2, Eso4 are "effects of species" [1/cm3]
  // -------------- h2o effects -------------------
  Real Eh2o = xkw;

  // -------------- co2 effects -------------------
  henry_factor_co2(t_factor,
                   // out
                   xk, xe);
  Real Eco2 = xk * xe * co2g * patm;

  // -------------- so4 effects -------------------
  // /cm3(a) (not sure which one this is in reference to)
  Real Eso4 = xso4 * xhnm * const0 / xlwc;

  /*
  -----------------------------------------------------------------
  now use bisection method to solve electro-neutrality equation
  to calculate PH value and H+ concentration

  during the iteration loop,
     yph_lo = lower pH value that brackets the root (i.e., correct pH)
     yph_hi = upper pH value that brackets the root (i.e., correct pH)
     yph    = current pH value
     yposnet_lo and yposnet_hi = net positive ions for
        yph_lo and yph_hi
  -----------------------------------------------------------------
  */

  constexpr Real zero = 0.0;
  converged = false;
  // ---------  1st iteration: set lower bound pH value ----------
  // FIXME: BAD CONSTANT
  Real yph_lo = 2.00;
  Real yph_hi = yph_lo;
  Real yph = yph_lo;
  Real ynetpos;
  calc_ynetpos(yph, fact1_so2, fact2_so2, fact3_so2, fact4_so2, Eco2, Eh2o,
               Eso4, so4_fact,
               // out
               xph, ynetpos);
  // FIXME: BAD CONSTANT
  if (ynetpos <= zero) {
    // the lower and upper bound pH values (2.0 and 7.0) do not bracket
    // the correct pH, so use the lower bound
    converged = true;
  }
  // FIXME: this appears to never be used
  // Real ynetpos_lo = ynetpos;

  // ---------  2nd iteration: set upper bound pH value ----------
  yph_hi = 7.0;
  yph = yph_hi;
  calc_ynetpos(yph, fact1_so2, fact2_so2, fact3_so2, fact4_so2, Eco2, Eh2o,
               Eso4, so4_fact,
               // out
               xph, ynetpos);
  // FIXME: BAD CONSTANT
  if (ynetpos >= zero) {
    // the lower and upper bound pH values (2.0 and 7.0) do not bracket
    // the correct pH, so use the lower bound
    converged = true;
  }
  // FIXME: this appears to never be used
  // Real ynetpos_hi = ynetpos;

  // --------- 3rd iteration and more ------------
  for (int i = 3; i < itermax; ++i) {
    yph = 0.5 * (yph_lo + yph_hi);
    calc_ynetpos(yph, fact1_so2, fact2_so2, fact3_so2, fact4_so2, Eco2, Eh2o,
                 Eso4, so4_fact,
                 // out
                 xph, ynetpos);
    if (ynetpos >= zero) {
      // net positive ions are >= 0 for both yph and yph_lo
      // so replace yph_lo with yph
      yph_lo = yph;
      // FIXME: this appears to never be used
      // ynetpos_lo = ynetpos;
    } else {
      // net positive ions are <= 0 for both yph and yph_hi
      // so replace yph_hi with yph
      yph_hi = yph;
      // FIXME: this appears to never be used
      // ynetpos_hi = ynetpos;
    }
    // FIXME: better error handling
    // FIXME: BAD CONSTANT
    if (mam4::abs(yph_hi - yph_lo) <= 0.005) {
      // |yph_hi - yph_lo| <= convergence criterion, so set
      // final pH to their midpoint and exit
      // (0.005 absolute error in pH gives 0.01 relative error in H+)
      yph = 0.5 * (yph_hi + yph_lo);
      xph = mam4::pow(10.0, -yph);
      converged = true;
      return;
    }
  } // end for(iter)
} // end calc_ph_values

//===========================================================================
KOKKOS_INLINE_FUNCTION
void calc_sox_aqueous(const bool modal_aerosols, const Real rah2o2,
                      const Real h2o2g, const Real so2g, const Real o3g,
                      const Real rao3, const Real patm, const Real dt,
                      const Real t_factor, const Real xlwc, const Real const0,
                      const Real xhnm, const Real heo3, const Real heso2,
                      // inout
                      Real &xso2, Real &xso4, Real &xso4_init, Real &xh2o2,
                      // out
                      Real &xdelso4hp) {
  /*
  -----------------------------------------------------------------
        ... Prediction after aqueous phase
        so4
        When Cloud is present

        S(IV) + H2O2 = S(VI)
        S(IV) + O3   = S(VI)

        reference:
            (1) Seinfeld
            (2) Benkovitz
  -----------------------------------------------------------------

    logical,  intent(in) :: modal_aerosols      // if using MAM
    real(r8), intent(in) :: rah2o2      // reaction rate with h2o2
    real(r8), intent(in) :: rao3        // reaction rate with o3
    real(r8), intent(in) :: h2o2g, so2g, o3g
    real(r8), intent(in) :: patm        // pressure [atm]
    real(r8), intent(in) :: dt       // time step [s]
    real(r8), intent(in) :: t_factor    // working variables to convert
    temperature real(r8), intent(in) :: xlwc        // in-cloud LWC [kg/L]
    real(r8), intent(in) :: const0
    real(r8), intent(in) :: xhnm
    real(r8), intent(in) :: heo3, heso2 // henry law constant
    real(r8), intent(inout) :: xso2, xso4, xso4_init, xh2o2 // mixing ratios
    real(r8), intent(out) :: xdelso4hp_ik // change of so4 in (i,k)

    // local variables
    real(r8) :: pso4    // production rate of so4
    real(r8) :: delta_s // so4 production in the time step
    real(r8) :: xk_so2, xk_h2o2 // for the use of Henry Laws
    real(r8) :: xe, x2  // output of henry law subroutines but not used
    real(r8), parameter :: small_value_20 = 1.0e-2 // small value
    real(r8), parameter :: small_value_30 = 1.0e-3 // small value

  ............................
        S(IV) + H2O2 = S(VI)
  ............................

  FORTRAN refactor: using henry law subroutines here break the BFB test
  because it changes the order of calculation. The original code is used now
  but new code is kept but commented out for future implementation.
    henry_factor_so2(t_factor, xk_so2, xe, x2)
    henry_factor_h2o2(t_factor, xk_h2o2, xe)
    pso4 = rah2o2 * xk_h2o2* h2o2g * patm * xk_so2 * so2g * patm
  */
  // FIXME: BAD CONSTANTS
  constexpr Real small_value_20 = 1.0e-20;
  constexpr Real small_value_30 = 1.0e-30;
  Real pso4 = rah2o2 * 7.4e4 * exp(6621.0 * t_factor) * h2o2g * patm * 1.23 *
              exp(3120.0 * t_factor) * so2g * patm;
  // [M/s] = [mole/L(w)/s] / [mole/L(a)/s] / [/L(a)/s]
  pso4 = pso4 * xlwc / const0 / xhnm;

  Real delta_s = mam4::max(pso4 * dt, small_value_30);

  xso4_init = xso4;

  if ((delta_s <= xso2) && (delta_s <= xh2o2)) {
    xso4 = xso4 + delta_s;
    xh2o2 = xh2o2 - delta_s;
    xso2 = xso2 - delta_s;
  } else if (xh2o2 > xso2) {
    xso4 = xso4 + xso2;
    xh2o2 = xh2o2 - xso2;
    xso2 = small_value_20;
  } else {
    xso4 = xso4 + xh2o2;
    xso2 = xso2 - xh2o2;
    xh2o2 = small_value_20;
  }

  if (modal_aerosols) {
    xdelso4hp = xso4 - xso4_init;
  }
  //...........................
  //       S(IV) + O3 = S(VI)
  //...........................

  // [M/s]
  pso4 = rao3 * heo3 * o3g * patm * heso2 * so2g * patm;
  // [M/s] =[mole/L(w)/s] * [mole/L(a)/s] / [/L(a)/s] / [mixing ratio/s]
  pso4 = pso4 * xlwc / const0 / xhnm;

  delta_s = mam4::max(pso4 * dt, small_value_30);

  xso4_init = xso4;

  if (delta_s > xso2) {
    xso4 = xso4 + xso2;
    xso2 = small_value_20;
  } else {
    xso4 = xso4 + delta_s;
    xso2 = xso2 - delta_s;
  }
} // end calc_sox_aqueous

// =============================================================================
KOKKOS_INLINE_FUNCTION
void compute_aer_factor(const Real *tmr, const int loffset,
                        const Config config_,
                        // out
                        Real faqgain_so4[AeroConfig::num_modes()]) {
  /*
  as called by sox_cldaero_update()
  compute_aer_factor(qcw, loffset, faqgain_so4)
  -------------------------------------------------------------------------
   compute factors for partitioning aerosol mass gains among modes
   the factors are proportional to the activated particle MR for each
   mode, which is the MR of cloud drops "associated with" the mode
   thus we are assuming the cloud drop size is independent of the
   associated aerosol mode properties (i.e., drops associated with
   Aitken and coarse sea-salt particles are same size)
   qnum_c(m) = activated particle number MR for mode n (these are just
   used for partitioning among modes, so don't need to divide by cldfrc)
  -------------------------------------------------------------------------
  real(r8), intent(in) :: tmr(:)    tracer mixing ratio [vmr]
  integer,  intent(in) :: loffset   # of tracers in the host model that are
  not part of MAM real(r8), intent(out) :: faqgain_so4(ntot_amode) factor
  of TMR among modes [fraction]

   local variables
  integer  :: m, ll                index
  real(r8) :: sumf                     total TMR for all modes [vmr]
  real(r8) :: qnum_c(ntot_amode)       tracer mixing ratio [vmr]
  real(r8), parameter :: small_value_10 = 1.e-10

  -------------------------------------------------------------------------
  */
  constexpr Real zero = 0.0;
  const int nmodes = AeroConfig::num_modes();

  Real sumf = zero;
  Real qnum_c[nmodes];
  for (int i = 0; i < nmodes; ++i) {
    qnum_c[i] = zero;
    faqgain_so4[i] = zero;
  }

  int ll;
  for (int m = 0; m < nmodes; ++m) {
    // NOTE: at least for the case of mam4, this might always be > 0
    ll = config_.numptrcw_amode[m] - loffset;
    // FIXME: I believe these two logic blocks can be combined
    if (ll > 0) {
      qnum_c[m] = mam4::max(zero, tmr[ll]);
    }
    // force qnum_c(m) to be positive for m = modeptr_accum or m = 1
    if (m == config_.modeptr_accum) {
      // FIXME: BAD CONSTANT
      qnum_c[m] = mam4::max(1.0e-10, qnum_c[m]);
    }

    // NOTE: given what I've seen for the value of lptr_so4_cw_amode in mam4,
    // this could probably be a loop
    // faqgain_so4(m) := fraction of total so4_c gain going to mode n
    // these are proportional to the activated particle MR for each mode
    if (config_.lptr_so4_cw_amode[m] > 0) {
      faqgain_so4[m] = qnum_c[m];
      sumf += faqgain_so4[m];
    }
  } // end for(nmodes)
  // at this point, (sumf <= 0.0) only when all the faqgain_so4 are zero
  if (sumf > zero) {
    for (int m = 0; m < AeroConfig::num_modes(); ++m) {
      faqgain_so4[m] /= sumf;
    }
  }

} // end compute_aer_factor

KOKKOS_INLINE_FUNCTION
Real cldaero_uptakerate(const Real xl, const Real cldnum, const Real cfact,
                        const Real cldfrc, const Real tfld, const Real press) {

  /*
  -----------------------------------------------------------------------
   compute uptake of h2so4 and msa to cloud water

   first-order uptake rate is
   4*pi*(drop radius)*(drop number conc)
   *(gas diffusivity)*(fuchs sutugin correction)
  -----------------------------------------------------------------------

  use mo_constants, only : pi

   input arguments
  real(r8), intent(in) :: xl           liquid water volume [cm^3/cm^3]
  real(r8), intent(in) :: cldnum       droplet number concentration
  [#/kg] real(r8), intent(in) :: cfact        total atms density [kg/L]
  real(r8), intent(in) :: cldfrc       cloud fraction [fraction]
  real(r8), intent(in) :: tfld         temperature [K]
  real(r8), intent(in) :: press        pressure [Pa]
   output arguments
  real(r8) :: uptkrate         uptake rate [cm/cm/s]
   local variables
  real(r8) :: rad_cd           droplet radius [cm]
  real(r8) :: radxnum_cd       radius * number conc [cm/cm^3]
  real(r8) :: num_cd           droplet number conc [1/cm^3]
  real(r8) :: gasdiffus        H2SO4 gas diffusivity [cm^2/s]
  real(r8) :: gasspeed         H2SO4 gas mean molecular speed [cm/s]
  real(r8) :: knudsen          knudsen number [unitless]
  real(r8) :: fuchs_sutugin    another dimensionless number
  real(r8) :: volx34pi_cd      droplet volume * 3/4 / pi [cm^3/cm^3]
  */

  // FIXME: (unforgivably) BAD CONSTANTS
  Real pi_x4 = 12.56637;
  Real one_third = 0.3333333;
  // conversion factor from cm^3 to L (or from 1/L to 1/cm^3)
  Real cm3_to_L = 1.0e-3;
  // artificial thresholds that assumes (radxnum_cd/volx34pi_cd < min)
  // and (radxnum_cd/volx34pi_cd > max) as unphysical
  Real min_factor_volx34pi_radxnum = 4.0e4;
  Real max_factor_volx34pi_radxnum = 4.0e8;

  //  change drop number conc from #/kg to #/cm^3
  Real num_cd = cm3_to_L * cldnum * cfact / cldfrc;
  num_cd = mam4::max(num_cd, 0.0);

  // (liquid water volume in cm^3/cm^3)
  Real volx34pi_cd = (xl * 0.75) / Constants::pi;
  // FIXME: this may not make sense--revisit this
  //  radxnum_cd = (drop radius)*(drop number conc)
  //  following holds because volx34pi_cd = num_cd*(rad_cd**3)
  Real radxnum_cd = mam4::pow(volx34pi_cd * square(num_cd), one_third);

  Real rad_cd;
  //  rad_cd = (drop radius in cm), computed from liquid water and drop number,
  //  then bounded by 0.5 and 50.0 micrometers to avoid the occasional
  //  unphysical value
  // FIXME: revisit this because weird
  if (radxnum_cd <= volx34pi_cd * min_factor_volx34pi_radxnum) {
    radxnum_cd = volx34pi_cd * min_factor_volx34pi_radxnum;
    rad_cd = 50.0e-4;
  } else if (radxnum_cd >= volx34pi_cd * max_factor_volx34pi_radxnum) {
    radxnum_cd = volx34pi_cd * max_factor_volx34pi_radxnum;
    rad_cd = 0.5e-4;
  } else {
    rad_cd = radxnum_cd / num_cd;
  }

  //  gasdiffus = h2so4 gas diffusivity from mosaic code (cm^2/s)
  //  (pmid must be Pa)
  // FIXME: BAD CONSTANTS
  Real gasdiffus = 0.557 * mam4::pow(tfld, 1.75) / press;

  //  gasspeed = h2so4 gas mean molecular speed from mosaic code (cm/s)
  // FIXME: BAD CONSTANTS
  Real gasspeed = 1.455e4 * mam4::sqrt(tfld / 98.0);

  //  knudsen number
  // FIXME(?): BAD CONSTANT(?)
  Real knudsen = 3.0 * gasdiffus / (gasspeed * rad_cd);

  /*
  following assumes accommodation coefficient accom=0.65
  (Adams & Seinfeld, 2002, JGR, and references therein)
  fuchs_sutugin = (0.75*accom*(1. + knudsen)) /
  (knudsen*(1.0 + knudsen + 0.283*accom) + 0.75*accom)
  */

  // FIXME: BAD CONSTANTS
  Real fuchs_sutugin =
      (0.4875 * (1.0 + knudsen)) / (knudsen * (1.184 + knudsen) + 0.4875);

  //  instantaneous uptake rate
  Real uptkrate = pi_x4 * radxnum_cd * gasdiffus * fuchs_sutugin;
  return uptkrate;
} // end function cldaero_uptakerate

KOKKOS_INLINE_FUNCTION
void update_tmr(Real &tmr, const Real dqdt, const Real dtime) {
  // -----------------------------------------------------------------------
  //  update tracer mixing ratio by adding tendencies
  // -----------------------------------------------------------------------
  // real(r8), intent(inout) :: tmr   tracer mixing ratio [vmr]
  // real(r8), intent(in)    :: dqdt  tmr tendency [vmr/s]
  // real(r8), intent(in)    :: dtime time step [s]

  tmr += dqdt * dtime;
} // end update_tmr

KOKKOS_INLINE_FUNCTION
void update_tmr_nonzero(Real &tmr, const int idx) {
  //-----------------------------------------------------------------------
  // this just makes sure the value is greater than zero
  // aka: max(a, tol)
  //-----------------------------------------------------------------------
  // tracer mixing ratio [vmr]
  // Real tmr;
  // index for the third dimension of vmr
  // int idx;

  // FIXME: BAD CONSTANT
  constexpr Real small_value_20 = 1.0e-20;

  // NOTE: in the fortran version, this if statement is if (idx > 0), so I
  // believe this is the correct way to port it
  if (idx >= 0) {
    tmr = mam4::max(tmr, small_value_20);
  }

} // end update_tmr_nonzero

//=================================================================================
KOKKOS_INLINE_FUNCTION
void sox_cldaero_update(const int loffset, const Real dt, const Real mbar,
                        const Real pdel, const Real press, const Real tfld,
                        const Real cldnum, const Real cldfrc, const Real cfact,
                        const Real xlwc, const Real delso4_hprxn,
                        const Real xh2so4, const Real xso4,
                        const Real xso4_init, const Config config_,
                        // inout
                        Real dqdt_aqso4[AeroConfig::num_gas_phase_species()],
                        Real dqdt_aqh2so4[AeroConfig::num_gas_phase_species()],
                        Real *qcw, Real *qin) {
  /*
  sox_cldaero_update(loffset, dt, mbar, pdel, press, tfld, cldnum,
                     cldfrc, cfact, cldconc.xlwc, xdelso4hp, xh2so4, xso4,
                     xso4_init,
                     // inout
                     qcw, qin);
  ----------------------------------------------------------------------------------
  Update the mixing ratios
  ----------------------------------------------------------------------------------

      // args
      integer,  intent(in) :: ncol
      integer,  intent(in) :: lchnk       // chunk id
      integer,  intent(in) :: loffset     // # of tracers in the host model
      that are not part of MAM real(r8), intent(in) :: dt       // time step
      [sec]

      real(r8), intent(in) :: mbar(:,:)   // mean wet atmospheric mass [amu
      or g/mol] real(r8), intent(in) :: pdel(:,:)   // pressure interval [Pa]
      real(r8), intent(in) :: press(:,:)  // pressure [Pa]
      real(r8), intent(in) :: tfld(:,:)   // temperature [K]

      real(r8), intent(in) :: cldnum(:,:) // droplet number concentration
      [#/kg] real(r8), intent(in) :: cldfrc(:,:) // cloud fraction [fraction]
      real(r8), intent(in) :: cfact(:,:)  // total atms density [kg/L]
      real(r8), intent(in) :: xlwc(:,:)   // liquid water volume [cm^3/cm^3]

      real(r8), intent(in) :: delso4_hprxn(:,:)   // change of so4 due to
      H2O2 chemistry [mol/mol] real(r8), intent(in) :: xh2so4(:,:)         //
      H2SO4 mass mixing ratio [mol/mol] real(r8), intent(in) :: xso4(:,:) //
      final SO4 mass mixing ratio [mol/mol] real(r8), intent(in) ::
      xso4_init(:,:)      // initial SO4 mass mixing ratio [mol/mol]

      real(r8), intent(inout) :: qcw(:,:,:) // cloud-borne aerosol [vmr]
      real(r8), intent(inout) :: qin(:,:,:) // xported species [vmr]

      // local vars ...

      // FORTRAN refactor note: aqueous chemistry (aq) here reprent two
      processes:
      //       S(IV) + H2O2 = S(VI)
      //       S(IV) + O3   = S(VI)
      // see the parent in mo_setsox for reference
  change of so4 due to O3 chemistry [mol/mol]
  real(r8) :: delso4_o3rxn
  so4_c tendency from aqueous chemistry [mol/mol/s]
  real(r8) :: dso4dt_aqrxn
  so4_c tendency from H2O2 chemistry [mol/mol/s]
  real(r8) :: dso4dt_hprxn
  so4_c tendency from h2so4 gas uptake [mol/mol/s]
  real(r8) :: dso4dt_gasuptk
  dqdt due to aqueous chemistry [mol/mol/s]
  real(r8) :: dqdt_aq
  dqdt due to wet removal, currently set as zero [mol/mol/s]
  real(r8) :: dqdt_wr
  dqdt due to so4 aqueous chemistry [mol/mol/s]
  real(r8) :: dqdt_aqso4(ncol,pver,gas_pcnst)
  dqdt due to h2so4 uptake [mol/mol/s]
  real(r8) :: dqdt_aqh2so4(ncol,pver,gas_pcnst)
  dqdt due to H2O2 chemistry [mol/mol/s]
  real(r8) :: dqdt_aqhprxn(ncol,pver)
  dqdt due to O3 chemistry [mol/mol/s]
  real(r8) :: dqdt_aqo3rxn(ncol,pver)
  integrated surface fluxes [kg/m2/s]
  real(r8) :: sflx(ncol)
  factor of TMR among modes [fraction]
  real(r8) :: faqgain_so4(ntot_amode)
  uptake rate [1/s]
  real(r8) :: uptkrate

      integer :: ll, mm, m    // aerosol mode index
      integer :: icol,kk          // column and level index
  */

  constexpr Real small_value_8 = 1.0e-8;
  constexpr Real small_value_5 = 1.0e-5;
  constexpr Real zero = 0.0;
  constexpr Real one = 1.0;

  const int nmodes = AeroConfig::num_modes();
  const int nspec_gas = AeroConfig::num_gas_phase_species();

  // make sure dqdt is zero initially, for budgets
  for (int i = 0; i < nspec_gas; ++i) {
    dqdt_aqso4[i] = zero;
    dqdt_aqh2so4[i] = zero;
  }
  // FIXME: these appear to never be used
  // Real dqdt_aqhprxn;
  // Real dqdt_aqo3rxn;

  if ((cldfrc >= small_value_5) && (xlwc >= small_value_8)) {
    /*
    -------------------------------------------------------------------------
    compute factors for partitioning aerosol mass gains among modes
    the factors are proportional to the activated particle MR for each
    mode, which is the MR of cloud drops "associated with" the mode
    thus we are assuming the cloud drop size is independent of the
    associated aerosol mode properties (i.e., drops associated with
    Aitken and coarse sea-salt particles are same size)
    */
    Real faqgain_so4[nmodes];
    compute_aer_factor(qcw, loffset, config_,
                       // out
                       faqgain_so4);

    Real uptkrate =
        cldaero_uptakerate(xlwc, cldnum, cfact, cldfrc, tfld, press);
    //   // average uptake rate over dt
    uptkrate = (one - mam4::exp(-one * mam4::min(100.0, dt * uptkrate))) / dt;
    //   // dso4dt_gasuptk = so4_c tendency from h2so4 gas uptake (mol/mol/s)
    Real dso4dt_gasuptk = xh2so4 * uptkrate;

    Real delso4_o3rxn = xso4 - xso4_init;
    Real dso4dt_aqrxn = (delso4_o3rxn + delso4_hprxn) / dt;
    Real dso4dt_hprxn = delso4_hprxn / dt;

    /*
    -----------------------------------------------------------------------
    now compute TMR tendencies
    this includes the above aqueous so2 chemistry AND
    the uptake of highly soluble aerosol precursor gases (h2so4, ...)
    The wetremoval of dissolved, unreacted so2 and h2o2 are assumed as zero
    */

    // dqdt due to aqueous chemistry [mol/mol/s]
    Real dqdt_aq;
    // dqdt due to wet removal, currently set as zero [mol/mol/s]
    // FIXME: this could probably be removed altogether, since it remains zero
    // throughout this function
    Real dqdt_wr = 0.0;
    // compute TMR tendencies for so4 aerosol-in-cloud-water
    // FIXME: there's a better way to do this
    for (int m = 0; m < nmodes; ++m) {
      int l = config_.lptr_so4_cw_amode[m] - loffset;
      if (l > 0) {
        dqdt_aqso4[l] = faqgain_so4[m] * dso4dt_aqrxn * cldfrc;
        dqdt_aqh2so4[l] = faqgain_so4[m] * dso4dt_gasuptk * cldfrc;
        dqdt_aq = dqdt_aqso4[l] + dqdt_aqh2so4[l];
        // don't have wet removal here
        // FIXME: maybe seems pointless for this to be a function?
        update_tmr(qcw[l], dqdt_aq + dqdt_wr, dt);
      }
    }

    /*
    For gas species, tendency includes reactive uptake to cloud water
    that essentially transforms the gas to a different species.
    Need to multiply both these parts by cldfrc
    Currently it assumes no wet removal here
    */

    // h2so4 (g)
    qin[config_.id_h2so4] =
        qin[config_.id_h2so4] - dso4dt_gasuptk * dt * cldfrc;
    // FORTRAN refactor: The order of multiplying cldfrc makes the following
    // call failing BFB test, so this calculation is not refactored with new
    // subroutine NOTE: I don't know what this refers to

    // so2 -- the first order loss rate for so2 is frso2_c*clwlrat(i,k)
    // don't have wet removal here
    dqdt_wr = 0.0;
    dqdt_aq = -dso4dt_aqrxn * cldfrc;
    update_tmr(qin[config_.id_so2], dqdt_aq + dqdt_wr, dt);

    // h2o2 -- the first order loss rate for h2o2 is frh2o2_c*clwlrat(i,k)
    // don't have wet removal here
    dqdt_wr = 0.0;
    dqdt_aq = -dso4dt_hprxn * cldfrc;
    update_tmr(qin[config_.id_h2o2], dqdt_aq + dqdt_wr, dt);

    /*
    for SO4 from H2O2/O3 budgets
    FIXME: these appear to never be used
    dqdt_aqhprxn = dso4dt_hprxn * cldfrc;
    dqdt_aqo3rxn = (dso4dt_aqrxn - dso4dt_hprxn) * cldfrc;
    */
  } // end if ((cldfrc >= small_value_5) && (xlwc >= small_value_8))

  //==============================================================
  // ... Update the mixing ratios
  //==============================================================
  // FIXME: this is very ugly. we loop through this array, 1 by 1, and if
  // (lptr_so4_cw_amode[m] - loffset) := idx > 0,
  // then it sets qcw to max(qcw, 1e-20)
  for (int m = 0; m < nmodes; ++m) {
    int tmp_idx = config_.lptr_so4_cw_amode[m] - loffset;
    // this is ugly, but better than passing the whole array--FIXME?
    if (0 <= tmp_idx)
      update_tmr_nonzero(qcw[tmp_idx], tmp_idx);
    // FIXME: I believe this is spurious.
    // as far as I can gather, lptr_nh4_cw_amode = [0, 0, 0, 0],
    // meaning that this results in a no-op
    // update_tmr_nonzero(qcw, (lptr_nh4_cw_amode[m] - loffset));
  }
  update_tmr_nonzero(qin[config_.id_so2], config_.id_so2);
} // end sox_cldaero_update

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION
void setsox_single_level(const int loffset, const Real dt, const Real press,
                         const Real pdel, const Real tfld, const Real mbar,
                         const Real lwc, const Real cldfrc, const Real cldnum,
                         const Real xhnm, Config setsox_config_,
                         // inout
                         Real dqdt_aqso4[AeroConfig::num_gas_phase_species()],
                         Real dqdt_aqh2so4[AeroConfig::num_gas_phase_species()],
                         Real qcw[AeroConfig::num_gas_phase_species()],
                         Real qin[AeroConfig::num_gas_phase_species()]) {

  // setsox_single_level(loffset, dt, press_k, pdel, tfld_k, mbar, lwc_k,
  //                           cldfrc_k, cldnum_k, xhnm_k, setsox_config_,
  //                           qcw_k, qin_k);
  /*
  don't know dimension of: mbar

  xnhm := total atms density [#/cm3]

  -----------------------------------------------------------------------
           ... Compute heterogeneous reactions of SOX

        (0) using initial PH to calculate PH
            (a) HENRYs law constants
            (b) PARTIONING
            (c) PH values

        (1) using new PH to repeat
            (a) HENRYs law constants
            (b) PARTIONING
            (c) REACTION rates
            (d) PREDICTION
  -----------------------------------------------------------------------

      use ppgrid,          only : pver
      use cam_history,     only : outfld
      use sox_cldaero_mod, only : sox_cldaero_update,
      sox_cldaero_create_obj, sox_cldaero_destroy_obj use cldaero_mod, only
      : cldaero_conc_t use cam_abortutils,  only : endrun


      implicit none

  -----------------------------------------------------------------------
       ... Dummy arguments
  -----------------------------------------------------------------------
      integer,          intent(in)    :: ncol              // num of
      columns in chunk integer,          intent(in)    :: lchnk // chunk id
      integer,          intent(in)    :: loffset           // offset of
      chem tracers in the advected tracers array real(r8), intent(in)    ::
      dt             // time step [sec] real(r8),         intent(in) ::
      press(:,:)        // midpoint pressure [Pa] real(r8), intent(in) ::
      pdel(:,:)         // pressure thickness of levels [Pa] real(r8),
      intent(in)    :: tfld(:,:)         // temperature [K] real(r8),
      intent(in)    :: mbar(:,:)         // mean wet atmospheric mass [amu
      or g/mol] real(r8), target, intent(in)    :: lwc(:,:)          //
      cloud liquid water content [kg/kg] real(r8), target, intent(in)    ::
      cldfrc(:,:)       // cloud fraction [fraction] real(r8), intent(in)
      :: cldnum(:,:)       // droplet number concentration [#/kg] real(r8),
      intent(in)    :: xhnm(:,:)         // total atms density [#/cm**3]
      real(r8), target, intent(inout) :: qcw(:,:,:)        // cloud-borne
      aerosol [vmr] real(r8),         intent(inout) :: qin(:,:,:)        //
      transported species [vmr]

  -----------------------------------------------------------------------
       ... Local variables

       FORTRAN refactoring: the units are a little messy here
       my understanding (may not be right) is that, the PH value xph, shown
       in [H+] concentration, is (mol H+)/(L water), which can be
       transfered to kg/L or kg/kg the variables xso2, xso4, xo3 etc have
       units of [mol/mol] (maybe corresponding to kg/kg above?) the
       variable xhnm has unit of [#/cm3]. Some units may changes to
       different formats across modules
       Shuaiqi Tang  4/18/2023
  -----------------------------------------------------------------------
      real(r8), parameter :: const0 = 1.0e30/6.023e230 // [cm3/L *
      mol/mole] real(r8), parameter :: Ra = 8314.0/101325.0 // universal
      constant   (atm)/(M-K) real(r8), parameter :: xkw = 1.0e-14 // water
      acidity real(r8), parameter :: p0 = 101300.0          // sea-level
      pressure [Pa] real(r8), parameter :: t298K = 298.0          //
      temperature of 25degC [K] real(r8), parameter :: small_value_lwc
      = 1.0e-8 // small value of LWC [kg/kg] real(r8), parameter ::
      small_value_cf = 1.0e-5  // small value of cloud fraction [fraction]

      integer  :: icol,kk
      logical  :: converged
      real(r8) :: t_factor       // working variables to convert
      temperature real(r8) :: cfact(ncol,pver)        // total atms density
      [kg/L] real(r8) :: xk, xe, x2     // output parameters in Henry's law
      real(r8) :: tz      // temperature at (i,k) [K]
      real(r8) :: xlwc    // in-cloud LWC at (i,k) [kg/L]
      real(r8) :: px      // temporary variable [unitless]
      real(r8) :: patm    // pressure [atm]

  the concentration values can vary in different forms
      real(r8) :: xph0,  xph(ncol,pver)   // pH value in H+ concentration
      [mol/L, or kg/L, or kg/kg(w)] real(r8) :: so2g, h2o2g, o3g        //
      concentration in gas phase [mol/mol] real(r8) :: rah2o2, rao3 //
      reaction rate
  mass concentration for species
      real(r8), dimension(ncol,pver) :: xso2, xso4, xso4_init, xh2so4, xo3,
      xh2o2
                                        // species concentrations [mol/mol]
      real(r8), pointer :: xso4c(:,:)
      real(r8) :: xdelso4hp(ncol,pver)    // change of so4 [mol/mol]
      real(r8) :: xphlwc(ncol,pver)       // pH value multiplied by lwc
      [kg/kg]

      real(r8), dimension(ncol,pver) :: heh2o2,heso2,heo3 // henry law
      const for species type(cldaero_conc_t), pointer :: cldconc
  #include "../yaml/mo_setsox/f90_yaml/setsox_beg_yml.f90"

  ==================================================================
        ... First set the PH
  ==================================================================
       ... Initial values
            The values of so2, so4 are after (1) SLT, and CHEM
  -----------------------------------------------------------------
  */

  constexpr Real one = 1.0;
  constexpr Real zero = 0.0;
  constexpr Real t298K = 298.0;

  constexpr Real ph0 = 5.0; // Initial PH values
  // initial PH value, in H+ concentration
  Real xph0 = mam4::pow(10, -ph0);
  // cfact := total atms density [kg/L]
  // FIXME: BAD CONSTANTS
  //           cm-3 * m-3    * kg/m3            * kg/L;
  Real cfact = xhnm * 1.0e6 * 1.38e-23 / 287.0 * 1.0e-3;

  // initialize species concentrations
  Cloudconc cldconc;
  // this name doesn't make sense after porting
  cldconc =
      sox_cldaero_create_obj(cldfrc, qcw, lwc, cfact, loffset, setsox_config_);

  /*
  if ( inv_so2 .or. id_hno3>0 .or. inv_h2o2 .or. id_nh3>0 .or. inv_o3 &
               .or. (.not. inv_ho2) .or. (.not. cloud_borne) .or.
               id_msa>0) then
      call endrun('FORTRAN refactoring: Only keep the code for default
      MAM4. &
           The following options are removed:  id_nh3>0  id_hno3>0
           id_msa>0 & inv_h2o2=.T. inv_so2=.T.  inv_o3=.T. inv_ho2=.F.
           cloud_borne=.F. ')
  endif
  */

  // species molar mixing ratios(?) [mol/mol]
  Real xso4 = zero;
  // initial PH value
  Real xph = xph0;
  Real xso2 = qin[setsox_config_.id_so2];
  Real xh2o2 = qin[setsox_config_.id_h2o2];
  Real xo3 = qin[setsox_config_.id_o3];
  Real xh2so4 = qin[setsox_config_.id_h2so4];

  /*
  -----------------------------------------------------------------
        ... Temperature dependent Henry constants
  -----------------------------------------------------------------
      ver_loop0: do kk = 1,pver          //// pver loop for STEP 0
         col_loop0: do icol = 1,ncol
  */
  // there doesn't appear to be any reason for doing this
  Real xso4c = cldconc.so4c;
  // in-cloud liquid water content

  Real xlwc = cldconc.xlwc;
  if (xlwc >= setsox_config_.small_value_lwc) {
    Real t_factor = (one / tfld) - (one / t298K);
    // calculate press in atm
    Real patm = press / setsox_config_.p0;

    if ((setsox_config_.cloud_borne > zero) && (cldfrc > zero)) {
      xso4 = xso4c / cldfrc;
    }
    bool converged = false;
    calc_ph_values(tfld, patm, xlwc, t_factor, xso2, xso4, xhnm,
                   cldconc.so4_fact, setsox_config_.Ra, setsox_config_.xkw,
                   setsox_config_.const0, setsox_config_.co2g,
                   setsox_config_.itermax,
                   // out
                   converged, xph);

    /*
    FIXME: better error handling
    if (!converged) {
    write(iulog, *) 'setsox: pH failed to converge @ (', icol, ',', kk,
    ').'
    }
    */
  } else {
    // FIXME: BAD CONSTANT
    xph = 1.0e-7;
  } // end if (xlwc >= small_value_xlwc)
  //==============================================================
  //          ... Now use the actual pH
  //==============================================================

  Real t_factor = (one / tfld) - (one / t298K);
  xlwc = cldconc.xlwc;
  // calculate press in atm
  Real patm = press / setsox_config_.p0;

  Real xk, xe, x2;

  //-----------------------------------------------------------------
  //        ... h2o2
  //-----------------------------------------------------------------
  henry_factor_h2o2(t_factor,
                    // out
                    xk, xe);
  Real heh2o2 = xk * (one + xe / xph);

  //-----------------------------------------------------------------
  //         ... so2
  //-----------------------------------------------------------------
  henry_factor_so2(t_factor,
                   // out
                   xk, xe, x2);
  Real heso2 = xk * (one + xe / xph * (one + x2 / xph));

  //-----------------------------------------------------------------
  //        ... o3
  //-----------------------------------------------------------------
  henry_factor_o3(t_factor,
                  // out
                  xk);
  Real heo3 = xk;

  //-----------------------------------------------
  //       ... Partioning
  //-----------------------------------------------
  Real tz = tfld;

  //------------------------------------------------------------------------
  //        ... h2o2
  //------------------------------------------------------------------------
  Real px = heh2o2 * setsox_config_.Ra * tz * xlwc;
  Real h2o2g = xh2o2 / (one + px);

  //------------------------------------------------------------------------
  //         ... so2
  //------------------------------------------------------------------------
  px = heso2 * setsox_config_.Ra * tz * xlwc;
  Real so2g = xso2 / (one + px);

  //------------------------------------------------------------------------
  //         ... o3
  //------------------------------------------------------------------------
  px = heo3 * setsox_config_.Ra * tz * xlwc;
  Real o3g = xo3 / (one + px);

  //-----------------------------------------------
  //       ... Aqueous phase reaction rates
  //           SO2 + H2O2 -> SO4
  //           SO2 + O3   -> SO4
  //-----------------------------------------------

  //------------------------------------------------------------------------
  //       ... S(IV) (HSO3) + H2O2
  //------------------------------------------------------------------------
  // A1: aqueous S(IV) + H2O2 -> S(VI). Reaction-family references:
  // Hoffmann and Edwards (1975), J. Phys. Chem. 79, 2096-2098,
  // doi:10.1021/j100587a005; Martin and Damschen (1981), Atmos. Environ.
  // 15, 1615-1621, doi:10.1016/0004-6981(81)90146-3; McArdle and Hoffmann
  // (1983), J. Phys. Chem. 87, 5425-5429, doi:10.1021/j150644a024.
  // These establish the kinetic family; the exact 8.e4, 3650, and 0.1 fit
  // below has not been independently matched to those papers.
  // xph is [H+] [mol L^-1]; t_factor = 1/T - 1/(298 K) [K^-1].
  // rah2o2 [L mol^-1 s^-1] multiplies dissolved molecular SO2 and H2O2
  // concentrations, using intrinsic Henry constants in calc_sox_aqueous.
  Real rah2o2 = 8.e4 * exp(-3650.0 * t_factor) / (0.1 + xph);

  //------------------------------------------------------------------------
  //        ... S(IV)+ O3
  //------------------------------------------------------------------------
  // A2: aqueous S(IV) + O3 -> S(VI). Reaction-family references:
  // Maahs (1983), Atmos. Environ. 17, 341-345,
  // doi:10.1016/0004-6981(83)90050-1; Hoffmann (1986), Atmos. Environ.
  // 20, 1145-1154, doi:10.1016/0004-6981(86)90147-2.
  // These establish the kinetic family; the exact two-term fit below has
  // not been independently matched to those papers.
  // tz is temperature [K]; xph is [H+] [mol L^-1]. rao3 [L mol^-1 s^-1]
  // multiplies dissolved O3 and S(IV) concentrations in calc_sox_aqueous,
  // using the effective SO2 Henry constant heso2 (including dissociation).
  Real rao3 = 4.39e11 * exp(-4131.0 / tz) + 2.56e3 * exp(-996.0 / tz) / xph;

  /*
  -----------------------------------------------------------------
        ... Prediction after aqueous phase
        so4
        When Cloud is present

        S(IV) + H2O2 = S(VI)
        S(IV) + O3   = S(VI)

        reference:
            (1) Seinfeld
            (2) Benkovitz
  -----------------------------------------------------------------
  */

  //............................
  //       S(IV) + H2O2 = S(VI)
  //............................

  Real xdelso4hp = 0, xso4_init = 0;
  // FIXME: in MAM4, xdelso4hp and xso4_init are uninitialized in the case
  // xlwc < small, and then used in sox_cldaero_update()
  // currently handling this by making it zero in the `else` case that does not
  // exist in the MAM4 fortran code
  // WHEN CLOUD IS PRESENTED (present?)
  if (xlwc >= setsox_config_.small_value_lwc) {
    calc_sox_aqueous(setsox_config_.modal_aerosols, rah2o2, h2o2g, so2g, o3g,
                     rao3, patm, dt, t_factor, xlwc, setsox_config_.const0,
                     xhnm, heo3, heso2,
                     // inout
                     xso2, xso4, xso4_init, xh2o2,
                     // out
                     xdelso4hp);
  } else {
    xdelso4hp = 0.0;
    xso4_init = 0.0;
  }

  // mean wet atmospheric mass [amu or g/mol]
  sox_cldaero_update(loffset, dt, mbar, pdel, press, tfld, cldnum, cldfrc,
                     cfact, cldconc.xlwc, xdelso4hp, xh2so4, xso4, xso4_init,
                     setsox_config_,
                     // inout
                     dqdt_aqso4, dqdt_aqh2so4, qcw, qin);
} //   end setsox_single_level

KOKKOS_INLINE_FUNCTION
void setsox(const ThreadTeam &team, const int loffset, const Real dt,
            const ColumnView &press, const ColumnView &pdel,
            const ColumnView &tfld, const ColumnView &mbar,
            const ColumnView &lwc, const ColumnView &cldfrc,
            const ColumnView &cldnum, const ColumnView &xhnm,
            // inout
            Real dqdt_aqso4[AeroConfig::num_gas_phase_species()],
            Real dqdt_aqh2so4[AeroConfig::num_gas_phase_species()],
            const ColumnView qcw[AeroConfig::num_gas_phase_species()],
            const ColumnView qin[AeroConfig::num_gas_phase_species()]) {

  const Config setsox_config_;
  constexpr int nk = mam4::nlev;

  // NOTE: pdel and mbar seem to be entirely unused and only used in mam4 to
  // calculate a quantity that is written out and otherwise unused
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nk), [&](int k) {
    // auto press = mam4::Atmosphere.pressure;
    const Real press_k = press(k);
    const Real pdel_k = pdel(k);
    const Real tfld_k = tfld(k);
    const Real mbar_k = mbar(k);
    const Real lwc_k = lwc(k);
    const Real cldfrc_k = cldfrc(k);
    const Real cldnum_k = cldnum(k);
    const Real xhnm_k = xhnm(k);
    const int nspec = AeroConfig::num_gas_phase_species();
    Real qcw_k[nspec];
    Real qin_k[nspec];
    for (int i = 0; i < nspec; ++i) {
      qcw_k[i] = qcw[i](k);
      qin_k[i] = qin[i](k);
    }
    setsox_single_level(loffset, dt, press_k, pdel_k, tfld_k, mbar_k, lwc_k,
                        cldfrc_k, cldnum_k, xhnm_k, setsox_config_, dqdt_aqso4,
                        dqdt_aqh2so4, qcw_k, qin_k);
  }); // end kokkos::parfor(k)

} // end setsox()

} // namespace mo_setsox
} // namespace mam4
#endif
