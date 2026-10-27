#ifndef MAM4XX_MO_SETHET_HPP
#define MAM4XX_MO_SETHET_HPP

#include "atmosphere.hpp"
#include "gas_chem_mechanism.hpp"
#include "mam4_constants.hpp"
#include "mam4_math.hpp"
#include <ekat_subview_utils.hpp>

namespace mam4 {

namespace mo_sethet {

constexpr Real avo = Constants::avogadro;
constexpr Real pi = Constants::pi;
constexpr Real rga = 1.0 / Constants::gravity;
constexpr int gas_pcnst = gas_chemistry::gas_pcnst;
const Real boltz_cgs = Constants::boltzmann * 1.e7; // erg/K
// number of vertical levels
constexpr int pver = mam4::nlev;

// FIXME: BAD CONSTANT
// mass of water vapor [amu] //convert to g/mol from kg/mol
constexpr Real mass_h2o = Constants::molec_weight_h2o * 1000;
constexpr Real cm3_2_m3 = 1.0e-6; // convert cm^3 to m^3
constexpr Real liter_per_gram = 1.0e-3;
constexpr Real avo2 =
    avo * liter_per_gram * cm3_2_m3; // [liter/gm/mol*(m/cm)^3]

using Real = Real;
using View1D = DeviceType::view_1d<Real>;
using View2D = DeviceType::view_2d<Real>;

//=================================================================================
KOKKOS_INLINE_FUNCTION
void calc_het_rates(const Real satf, // saturation fraction in cloud //in
                    const Real rain, // rain rate [molecules/cm^3/s] //in
                    const Real xhen, // henry's law constant // in
                    const Real tmp_hetrates, const Real work1,
                    const Real work2, // in
                    Real &het_rates)  // rainout loss rates [1/s]// out
{
  //-----------------------------------------------------------------
  // calculate het_rates
  // input arguments are different for different species
  //-----------------------------------------------------------------
  Real work3;
  Real h2o_mol = 1.0e3 / mass_h2o; // [gm/mol water]

  work3 =
      satf * mam4::max(rain / (h2o_mol * (work1 + 1.0 / (xhen * work2))), 0.0);
  het_rates = work3 + tmp_hetrates;

} // end calc_het_rates
//=================================================================================

//=================================================================================
KOKKOS_INLINE_FUNCTION
void calc_precip_rescale(
    const ThreadTeam &team,
    const ColumnView &cmfdqr,      // dq/dt for convection [kg/kg/s] //in
    const ConstColumnView &nrain,  // stratoform precip [kg/kg/s] //in
    const ConstColumnView &nevapr, // evaporation [kg/kg/s] // in
    const ColumnView &precip) {    // precipitation [kg/kg/s] // out
  // -----------------------------------------------------------------------
  // calculate precipitation rate at each grid
  // this is added to rescale the variable precip (which can only be positive)
  // to the actual vertical integral of positive and negative values.
  // This removes point storms
  // -----------------------------------------------------------------------

  Real total_rain = 0; // total rain rate (both pos and neg) in the column
  Real total_pos = 0;  // total positive rain rate in the column
  constexpr int local_pver = pver;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, local_pver),
      [&](int kk, Real &total_rain) {
        precip(kk) = cmfdqr(kk) + nrain(kk) - nevapr(kk);
        total_rain += precip(kk);
      },
      total_rain);
  team.team_barrier();
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, local_pver),
      [&](int kk, Real &total_pos) {
        if (precip(kk) < 0.0) {
          precip(kk) = 0.0;
        }
        total_pos += precip(kk);
      },
      total_pos);
  team.team_barrier();
  if (total_rain <= 0.0) {
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, local_pver),
                         [&](int kk) {
                           precip(kk) = 0.0; // set all levels to zero
                         });
  } else {
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team, local_pver),
        [&](int kk) { precip(kk) = precip(kk) * total_rain / total_pos; });
  }
} // end subroutine calc_precip_rescale

//=================================================================================
KOKKOS_INLINE_FUNCTION
void gas_washout(
    const ThreadTeam &team, const int ktop, const int pver,
    const Real xkgm,       // mass flux on rain drop //in
    const ColumnView xliq, // liquid rain water content [gm/m^3] // in
    const ColumnView rain,
    const ColumnView xhen_i,      // henry's law constant
    const ConstColumnView tfld_i, // temperature [K]
    const ColumnView delz_i,      // layer depth about interfaces [cm]  // in
    const ColumnView xgas,        // gas concentration // inout
    const ColumnView scratch) {
  //------------------------------------------------------------------------
  // calculate gas washout by cloud if not saturated
  //------------------------------------------------------------------------
  // FIXME: BAD CONSTANTS
  const Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
  constexpr Real geo_fac =
      6.0; // geometry factor (surf area/volume = geo_fac/diameter)
  constexpr Real xrm = .189;  // mean diameter of rain drop [cm]
  constexpr Real xum = 748.0; // mean rain drop terminal velocity [cm/s]

  // -----------------------------------------------------------------
  //       ... calculate the saturation concentration eqca
  // -----------------------------------------------------------------
  // total of ca between level plev and kk [#/cm3]
  for (int kk = ktop; kk < pver; ++kk) {
    if (rain(kk) != 0.0) { // finding rain cloud
      Kokkos::parallel_scan(
          Kokkos::TeamThreadRange(team, kk, pver),
          [&](const int k, Real &accumulator, const bool last) {
            const Real xca = geo_fac * xkgm * xgas(k) / (xrm * xum) *
                             delz_i(k) * xliq(kk) * cm3_2_m3;
            accumulator += xca;
            if (last)
              scratch[k] = accumulator;
          });
      team.team_barrier();
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(team, kk, pver), [&](const int k) {
            const Real xeqca =
                xgas(k) /
                (xliq(kk) * avo2 + 1.0 / (xhen_i(k) * const0 * tfld_i(k))) *
                xliq(kk) * avo2;

            // -----------------------------------------------------------------
            //       ... if is not saturated (take hno3 as an example)
            //               hno3(gas)_new = hno3(gas)_old - hno3(h2o)
            //           otherwise
            //               hno3(gas)_new = hno3(gas)_old
            // -----------------------------------------------------------------
            if (scratch[k] < xeqca) {
              //-----------------------------------------------------------------
              //       ... calculate ca; inside cloud concentration in
              //       #/cm3(air)
              //-----------------------------------------------------------------
              const Real xca = geo_fac * xkgm * xgas(k) / (xrm * xum) *
                               delz_i(k) * xliq(kk) * cm3_2_m3;
              xgas(k) = mam4::max(xgas(k) - xca, 0.0);
            }
          });
      team.team_barrier();
    }
  }
} // end subroutine gas_washout

//=================================================================================
KOKKOS_INLINE_FUNCTION
void find_ktop(
    Real rlat,                    // latitude in radians for columns
    const ConstColumnView &press, // pressure [Pa] // in
    int &ktop) { // index that only calculate het_rates above this level //out
  //---------------------------------------------------------------------------
  // -------- find the top level that het_rates are set as 0 above it ---------
  //---------------------------------------------------------------------------

  Real p_limit = 0;      // pressure limit [Pa]
  Real d2r = pi / 180.0; // degree to radian

  if (mam4::abs(rlat) > 60.0 * d2r) {
    p_limit = 300.0e2; // 300hPa for high latitudes
  } else {
    p_limit = 100.0e2; // 100hPa for low latitudes
  }

  for (int kk = pver - 1; kk >= 0; kk--) {
    if (press(kk) < p_limit) {
      ktop = kk;
      return;
    }
  } // k_loop

} // end subroutine find_ktop

KOKKOS_INLINE_FUNCTION
constexpr int get_work_len_sethet() {
  // work_len these variables:
  // t_factor, xk0_hno3, xk0_so2, so2_diss, xgas2, xgas3,
  // delz, xh2o2, xso2, xliq, rain, precip,
  // xhen_h2o2, xhen_hno3, xhen_so2, tmp_hetrates
  return 15 * nlev + gas_pcnst * nlev;
}

KOKKOS_INLINE_FUNCTION
int get_total_work_len_sethet() {
  // work_len these variables:
  // including het_rates, dqdt_aqso4, dqdt_aqh2so4 and vmr_col
  const int sether_work_len = get_work_len_sethet();
  return sether_work_len + 3 * (nlev * gas_pcnst) + (gas_pcnst * nlev);
}

KOKKOS_INLINE_FUNCTION
void sethet_detail(
    const ThreadTeam &team,
    const View2D &het_rates,       // rainout rates [1/s] //out
    const Real rlat,               // latitude in radians for columns
    const ConstColumnView &press,  // pressure [pascals] //in
    const ConstColumnView &zmid,   // midpoint geopot [km]  //in
    const Real phis,               // surf geopotential //in
    const ConstColumnView &tfld,   // temperature [K]  //in
    const ColumnView &cmfdqr,      // dq/dt for convection [kg/kg/s] //in
    const ConstColumnView &nrain,  // stratoform precip [kg/kg/s] //in
    const ConstColumnView &nevapr, // evaporation [kg/kg/s] //in
    const Real delt,               // time step [s] //in
    const View2D &invariants,      // total atms density [cm^-3] //in
    const View2D &qin,             // xported species [vmr]  //in
    // working variables
    const ColumnView
        &t_factor, // temperature factor to calculate henry's law parameters
    const ColumnView &xk0_hno3, const ColumnView &xk0_so2,
    const ColumnView &so2_diss, // so2 dissociation constant
    const ColumnView
        &xgas2, // gas phase species for h2o2 (2) and so2 (3) [molecules/cm^3]
    const ColumnView
        &xgas3, // gas phase species for h2o2 (2) and so2 (3) [molecules/cm^3]
    const ColumnView &delz,  // layer depth about interfaces [cm]
    const ColumnView &xh2o2, // h2o2 concentration [molecules/cm^3]
    const ColumnView &xso2,  // so2 concentration [molecules/cm^3]
    const ColumnView &xliq, // liquid rain water content in a grid cell [gm/m^3]
    const ColumnView &rain, // precipitation (rain) rate [molecules/cm^3/s]
    const ColumnView &precip,    // precipitation rate [kg/kg/s]
    const ColumnView &xhen_h2o2, // henry law constants
    const ColumnView &xhen_hno3, // henry law constants
    const ColumnView &xhen_so2,  // henry law constants
    const View2D tmp_hetrates, const int spc_h2o2_ndx, const int spc_so2_ndx,
    const int h2o2_ndx, const int so2_ndx, const int h2so4_ndx,
    const int gas_wetdep_cnt, const int wetdep_map[3], const int indexm) {



  const int pver = mam4::nlev;
  //-----------------------------------------------------------------------
  //       ... compute rainout loss rates (1/s)
  //-----------------------------------------------------------------------
  constexpr int local_pver = pver;
  //-----------------------------------------------------------------------
  //       ... local variables       //FIXME: BAD CONSTANT
  //-----------------------------------------------------------------------
  constexpr Real xrm = .189;       // mean diameter of rain drop [cm]
  constexpr Real xum = 748.0;      // mean rain drop terminal velocity [cm/s]
  constexpr Real xvv = 6.18e-2;    // kinetic viscosity [cm^2/s]
  constexpr Real xdg = .112;       // mass transport coefficient [cm/s]
  constexpr Real t0 = 298.0;       // reference temperature [K]
  constexpr Real xph0 = 1.0e-5;    // cloud [h+]
  constexpr Real satf_hno3 = .016; // saturation factor for hno3 in clouds
  constexpr Real satf_h2o2 = .016; // saturation factor for h2o2 in clouds
  constexpr Real satf_so2 = .016;  // saturation factor for so2 in clouds
  const Real const0 = boltz_cgs * 1.0e-6; // [atmospheres/deg k/cm^3]
  constexpr Real hno3_diss = 15.4;        // hno3 dissociation constant
  constexpr Real mass_air = 29.0;         // mass of background atmosphere [amu]
  constexpr Real km2cm = 1.0e5;           // convert km to cm
  constexpr Real m2km = 1.0e-3;           // convert m to km
  constexpr Real m3_2_cm3 = 1.0e6;        // convert m^3 to cm^3
  constexpr Real MISSING = -999999.0;
  constexpr Real large_value_lifetime =
      1.0e29; // a large lifetime value if no washout

  int ktop = 0;  // tropopause level, 100mb for lat < 60 and 300mb for lat > 60
  Real xkgm = 0; // mass flux on rain drop
  Real work1 = 0, work2 = 0; // working variables
  Real zsurf = 0;            // surface height [km]

  //-----------------------------------------------------------------
  //        note: the press array is in pascals and must be
  //              mutiplied by 10 to yield dynes/cm**2.
  //-----------------------------------------------------------------
  //       ... set wet deposition for
  //           1. h2o2         2. hno3
  //           3. ch2o         4. ch3ooh
  //           5. pooh         6. ch3coooh
  //           7. ho2no2       8. onit
  //           9. mvk         10. macr
  //          11. c2h5ooh     12. c3h7ooh
  //          13. rooh        14. ch3cocho
  //          15. pb          16. macrooh
  //          17. xooh        18. onitr
  //          19. isopooh     20. ch3oh
  //          21. c2h5oh      22. glyald
  //          23. hyac        24. hydrald
  //          25. ch3cho      26. isopno3
  //-----------------------------------------------------------------
  // FORTRAN refactor note: current MAM4 only have three species in default:
  // 'H2O2','H2SO4','SO2'.  Options for other species are then removed
  //-----------------------------------------------------------------

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, local_pver), [&](int kk) {
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      het_rates(kk, mm) = 0.0;
      tmp_hetrates(mm, kk) = 0.0; // initiate temporary array
    }
  });
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, local_pver), [&](int kk) {
    for (int mm = 0; mm < gas_wetdep_cnt; mm++) {
      const int mm2 = wetdep_map[mm];
      if (mm2 >= 0)
        het_rates(kk, mm2) = MISSING;
    }
  });
  team.team_barrier();
  //-----------------------------------------------------------------
  //	... the 2 and .6 multipliers are from a formula by frossling (1938)
  //-----------------------------------------------------------------
  xkgm = xdg / xrm * 2.0 + xdg / xrm * .6 * mam4::sqrt(xrm * xum / xvv) *
                               mam4::pow((xvv / xdg), (1.0 / 3.0));

  //-----------------------------------------------------------------
  //	... Find the level index that only calculate het_rates below
  //-----------------------------------------------------------------
  find_ktop(rlat, press, ktop); // populate ktop

  // this is added to rescale the variable precip (which can only be positive)
  // to the actual vertical integral of positive and negative values.  This
  // removes point storms
  calc_precip_rescale(team, cmfdqr, nrain, nevapr, precip); // populate precip
  team.team_barrier();

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, local_pver), [&](int kk) {
    rain(kk) = mass_air * precip(kk) * invariants(kk, indexm) / mass_h2o;
    xliq(kk) =
        precip(kk) * delt * invariants(kk, indexm) / avo * mass_air * m3_2_cm3;
    xh2o2(kk) = qin(kk, spc_h2o2_ndx) * invariants(kk, indexm);
    xso2(kk) = qin(kk, spc_so2_ndx) * invariants(kk, indexm);
  });
  zsurf = m2km * phis * rga;

  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, ktop, local_pver - 1),
      [&](int kk) { delz(kk) = mam4::abs((zmid(kk) - zmid(kk + 1)) * km2cm); });
  delz(pver - 1) = mam4::abs((zmid(pver - 1) - zsurf) * km2cm);
  //-----------------------------------------------------------------
  //       ... part 0b,  for temperature dependent of henrys
  //                     xxhe1 = henry con for hno3
  //                     xxhe2 = henry con for h2o2
  // lwh 10/00 -- take henry''s law constants from brasseur et al. [1999],
  //             appendix j. for hno3, also consider dissociation to
  //             get effective henry''s law constant; equilibrium
  //             constant for dissociation from brasseur et al. [1999],
  //             appendix k. assume ph=5 (set as xph0 above).
  //             heff = h*k/[h+] for hno3 (complete dissociation)
  //             heff = h for h2o2 (no dissociation)
  //             heff = h * (1 + k/[h+]) (in general)
  //-----------------------------------------------------------------
  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, ktop, local_pver), [&](int kk) {
        //-----------------------------------------------------------------
        // 	... effective henry''s law constants:
        //	hno3, h2o2  (brasseur et al., 1999)
        //-----------------------------------------------------------------
        // temperature factor
        t_factor(kk) = (t0 - tfld(kk)) / (t0 * tfld(kk));
        xhen_h2o2(kk) = 7.45e4 * mam4::exp(6620.0 * t_factor(kk));
        // HNO3, for calculation of H2SO4 het rate use
        xk0_hno3(kk) = 2.1e5 * mam4::exp(8700.0 * t_factor(kk));
        xhen_hno3(kk) = xk0_hno3(kk) * (1.0 + hno3_diss / xph0);
        // SO2
        xk0_so2(kk) = 1.23 * mam4::exp(3120.0 * t_factor(kk));
        so2_diss(kk) = 1.23e-2 * mam4::exp(1960.0 * t_factor(kk));
        xhen_so2(kk) = xk0_so2(kk) * (1.0 + so2_diss(kk) / xph0);
      });
  //-----------------------------------------------------------------
  //       ... part 1, solve for high henry constant ( hno3, h2o2)
  //-----------------------------------------------------------------
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, local_pver), [&](int kk) {
    xgas2(kk) = xh2o2(kk); // different levels wash
    xgas3(kk) = xso2(kk);
  });
  team.team_barrier();
  // gas_washout is odd in that it modifies all of xgas2 and xgas3
  // from level kk to level pver to calculate gas washout by cloud
  View1D scratch_space = ekat::subview(tmp_hetrates, 0);
  gas_washout(team, ktop, pver, xkgm, xliq, rain, // in
              xhen_h2o2, tfld, delz,              // in
              xgas2, scratch_space);              // inout
  gas_washout(team, ktop, pver, xkgm, xliq, rain, // in
              xhen_so2, tfld, delz,               // in
              xgas3, scratch_space);              // inout
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, local_pver),
                       [&](int kk) { scratch_space(kk) = 0.0; });
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, ktop, pver), [&](int kk) {
    Real stay =
        1.0; // fraction of layer traversed by falling drop in timestep delt
    if (rain(kk) != 0.0) { // finding rain cloud
      stay = ((zmid(kk) - zsurf) * km2cm) / (xum * delt);
      stay = mam4::min(stay, 1.0);
    }
    //-----------------------------------------------------------------
    //       ... calculate the lifetime of washout (second)
    //             after all layers washout
    //             the concentration of hno3 is reduced
    //             then the lifetime xtt is calculated by
    //
    //                  xtt = (xhno3(ini) - xgas1(new))/(dt*xhno3(ini))
    //                  where dt = passing time (s) in vertical
    //                             path below the cloud
    //                        dt = dz(cm)/um(cm/s)
    //-----------------------------------------------------------------
    const Real xdtm = delz(kk) / xum; // the traveling time in each dz

    const Real xxx2 = (xh2o2(kk) - xgas2(kk));
    Real yh2o2 = 0;    // washout lifetime [s]
    if (xxx2 != 0.0) { // if no washout lifetime = 1.e29
      yh2o2 = xh2o2(kk) / xxx2 * xdtm;
    } else {
      yh2o2 = large_value_lifetime;
    }
    tmp_hetrates(1, kk) =
        mam4::max(1.0 / yh2o2, 0.0) * stay; // FIXME: bad constant index

    Real yso2 = 0;
    const Real xxx3 =
        (xso2(kk) - xgas3(kk)); // working variables for h2o2 (2) and so2 (3)
    if (xxx3 != 0.0) {          // if no washout lifetime = 1.e29
      yso2 = xso2(kk) / xxx3 * xdtm;
    } else {
      yso2 = large_value_lifetime;
    }
    tmp_hetrates(2, kk) =
        mam4::max(1.0 / yso2, 0.0) * stay; // FIXME: bad constant index
  });
  team.team_barrier();
  //-----------------------------------------------------------------
  //       ... part 2, in-cloud solve for low henry constant
  //                   hno3 and h2o2 have both in and under cloud
  //-----------------------------------------------------------------
  constexpr int local_gas_pcnst = gas_pcnst;
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, ktop, pver), [&](int kk) {
    bool skip = false;
    for (int mm = 0; mm < local_gas_pcnst; ++mm) {
      if (rain(kk) <= 0.0) {
        het_rates(kk, mm) = 0.0;
        skip = true;
      }
    }
    if (!skip) {

      work1 = avo2 * xliq(kk);
      work2 = const0 * tfld(kk);

      if (h2o2_ndx >= 0) {
        calc_het_rates(satf_h2o2, rain(kk), xhen_h2o2(kk), // in
                       tmp_hetrates(1, kk), work1, work2,  // in
                       het_rates(kk, h2o2_ndx));           // out
      }
      // if ( prog_modal_aero .and.
      if (so2_ndx >= 0 && h2o2_ndx >= 0) {
        het_rates(kk, so2_ndx) = het_rates(kk, h2o2_ndx);
      } else if (so2_ndx >= 0) {
        calc_het_rates(satf_so2, rain(kk), xhen_so2(kk),  // in
                       tmp_hetrates(2, kk), work1, work2, // in
                       het_rates(kk, so2_ndx));           // out
      }

      if (h2so4_ndx >= 0) {
        calc_het_rates(satf_hno3, rain(kk), xhen_hno3(kk), // in
                       tmp_hetrates(0, kk), work1, work2,  // in
                       het_rates(kk, h2so4_ndx));          // out
      }
    }
  });
  team.team_barrier();
  //-----------------------------------------------------------------
  //	... Set rates above tropopause = 0.
  //-----------------------------------------------------------------
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, ktop + 1), [&](int kk) {
    for (int mm = 0; mm < gas_wetdep_cnt; mm++)
      het_rates(kk, wetdep_map[mm]) = 0.0;
  });
  team.team_barrier();
  int abort = 0;
  for (int mm = 0; mm < gas_wetdep_cnt; mm++)
    for (int kk = 0; kk < local_pver; ++kk)
      if (het_rates(kk, wetdep_map[mm]) == MISSING)
        ++abort;
  if (0 < abort)
    Kokkos::abort(
        "sethet: het_rates (wet dep) not set for het reaction number");
} // end subroutine sethet

KOKKOS_INLINE_FUNCTION
void sethet(
    const ThreadTeam &team, const Atmosphere &atm,
    const View2D &het_rates,      //[pver][gas_pcnst], rainout rates [1/s] //out
    const Real rlat,              // latitude in radians for columns
    const Real phis,              // surf geopotential //in
    const ColumnView &cmfdqr,     // dq/dt for convection [kg/kg/s] //in
    const ConstColumnView &prain, // stratoform precip [kg/kg/s] //in
    const ConstColumnView &nevapr, // evaporation [kg/kg/s] //in
    const Real dt,                 // time step [s] //in
    const View2D &invariants,      //
    const View2D &vmr,             // xported species [vmr]  //in
    // working variables
    const View1D &work) {

  auto work_ptr = (Real *)work.data();
  const auto t_factor = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xk0_hno3 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xk0_so2 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto so2_diss = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xgas2 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xgas3 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto delz = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xh2o2 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xso2 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xliq = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto rain = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto precip = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xhen_h2o2 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xhen_hno3 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  const auto xhen_so2 = View1D(work_ptr, nlev);
  work_ptr += nlev;
  View2D tmp_hetrates = View2D(work_ptr, gas_pcnst, nlev);
  work_ptr += gas_pcnst * nlev;

  // BAD CONSTANT
  // FIXME: should we move these indices and map to a config file?
  constexpr int spc_h2o2_ndx = 1;   // spc_h2o2_ndx = get_spc_ndx( 'H2O2' )
  constexpr int spc_so2_ndx = 3;    // spc_so2_ndx  = get_spc_ndx( 'SO2' )
  constexpr int h2o2_ndx = 1;       // h2o2_ndx   = get_het_ndx( 'H2O2' )
  constexpr int so2_ndx = 3;        // so2_ndx     = get_het_ndx( 'SO2' )
  constexpr int h2so4_ndx = 2;      // h2so4_ndx   = get_het_ndx( 'H2SO4' )
  constexpr int gas_wetdep_cnt = 3; // integer
  constexpr int wetdep_map[3] = {1, 2, 3};

  // index of total atm density in invariant array
  constexpr int indexm = mam4::gas_chemistry::indexm;

  mo_sethet::sethet_detail(team, het_rates, rlat, atm.pressure,
                           atm.height, // zmid,
                           phis, atm.temperature,
                           cmfdqr,     //
                           prain,      //
                           nevapr,     //
                           dt,         //
                           invariants, //
                           vmr,        //
                           t_factor,   //
                           xk0_hno3,   //
                           xk0_so2,    //
                           so2_diss,   //
                           xgas2,      //
                           xgas3,      //
                           delz, xh2o2, xso2, xliq, rain, precip, xhen_h2o2,
                           xhen_hno3, xhen_so2, tmp_hetrates, spc_h2o2_ndx,
                           spc_so2_ndx, h2o2_ndx, so2_ndx, h2so4_ndx,
                           gas_wetdep_cnt, wetdep_map, indexm);
}
} // namespace mo_sethet
} // namespace mam4
#endif
