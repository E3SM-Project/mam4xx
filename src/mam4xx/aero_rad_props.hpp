#ifndef MAM4XX_AERO_RAD_PROPS_HPP
#define MAM4XX_AERO_RAD_PROPS_HPP

#include "aero_config.hpp"

#include "modal_aero_opt.hpp"
#include "tropopause.hpp"

namespace mam4 {

namespace aero_rad_props {

// From radconstants
constexpr int nswbands = modal_aero_opt::nswbands;
constexpr int nlwbands = modal_aero_opt::nlwbands;
using View2D = DeviceType::view_2d<Real>;
using namespace mam4::modal_aero_opt;
constexpr Real km_inv_to_m_inv = 0.001; // 1/km to 1/m

constexpr Real shr_const_rgas =
    Constants::r_gas * 1e3; // Universal gas constant ~ J/K/kmole
constexpr Real shr_const_mwdair = Constants::molec_weight_dry_air *
                                  1e3; // molecular weight dry air ~ kg/kmole
constexpr Real shr_const_cpdair =
    Constants::cp_dry_air; // specific heat of dry air   ~ J/kg/K

constexpr Real cnst_kap =
    (shr_const_rgas / shr_const_mwdair) / shr_const_cpdair; //   R/Cp

constexpr Real cnst_faktor =
    -Constants::gravity /
    Constants::r_gas_dry_air; // acceleration of gravity ~ m/s^2/Dry air
                              // gas constant     ~ J/K/kg
constexpr Real cnst_ka1 = cnst_kap - 1.0;

// Similar to volcanic_cmip_sw,
// But, tau, tau_w, tau_w_g, tau_w_f have layout (nswbands, pver)
KOKKOS_INLINE_FUNCTION
void volcanic_cmip_sw(const ThreadTeam &team, const ConstColumnView &zi,
                      const int ilev_tropp, const View2D &ext_cmip6_sw_inv_m,
                      const View2D &ssa_cmip6_sw, const View2D &af_cmip6_sw,
                      const View2D &tau, const View2D &tau_w,
                      const View2D &tau_w_g, const View2D &tau_w_f) {

  /*
    zi(:,:)     Height above surface at interfaces [m]
    trop_level(pcols)   tropopause level index
    ext_cmip6_sw_inv_m(pcols,pver,nswbands)   short wave extinction [m^{-1}]
    ssa_cmip6_sw(:,:,:),af_cmip6_sw(:,:,:)

   Intent-inout
   Note: we are using emaxx layouts
    tau    (pcols,nswbands,0:pver)  aerosol extinction optical depth
    tau_w  (pcols,nswbands,0:pver)  aerosol single scattering albedo * tau
    tau_w_g(pcols,nswbands,0:pver)  aerosol assymetry parameter * tau * w
    tau_w_f(pcols,nswbands,0:pver)  aerosol forward scattered fraction * tau
    *
    w

   Local variables
   icol, ipver, ilev_tropp
   lyr_thk  thickness between level interfaces [m]
   ext_unitless(nswbands), asym_unitless(nswbands)
   ext_ssa(nswbands),ext_ssa_asym(nswbands)

   Logic:
   Update taus, tau_w, tau_w_g and tau_w_f with the read in volcanic
   aerosol extinction (1/km), single scattering albedo and asymmtry factors.

   Above the tropopause, the read in values from the file include both the
   stratospheric and volcanic aerosols. Therefore, we need to zero out taus
   above the tropopause and populate them exclusively from the read in
   values.

   If tropopause is found, update taus with 50% contributuions from the
   volcanic input file and 50% from the existing model computed values

   First handle the case of tropopause layer itself:
   do icol = 1, ncol
   ilev_tropp = trop_level(icol) tropopause level
  */

  constexpr Real half = 0.5;
  const Real lyr_thk = zi(ilev_tropp) - zi(ilev_tropp + 1);
  static constexpr int nswbands_loc = nswbands;
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nswbands_loc), [&](int i) {
    // NOTE: shape of ext_cmip6_sw_inv_m (nswbands,pver)
    const Real ext_unitless = lyr_thk * ext_cmip6_sw_inv_m(i, ilev_tropp);
    const Real asym_unitless = af_cmip6_sw(ilev_tropp, i);
    const Real ext_ssa = ext_unitless * ssa_cmip6_sw(ilev_tropp, i);
    const Real ext_ssa_asym = ext_ssa * asym_unitless;
    // NOTE: tau vars have one extra dimension in Fortran.
    tau(i, ilev_tropp + 1) = half * (tau(i, ilev_tropp + 1) + ext_unitless);
    tau_w(i, ilev_tropp + 1) = half * (tau_w(i, ilev_tropp + 1) + ext_ssa);
    tau_w_g(i, ilev_tropp + 1) =
        half * (tau_w_g(i, ilev_tropp + 1) + ext_ssa_asym);
    tau_w_f(i, ilev_tropp + 1) =
        half * (tau_w_f(i, ilev_tropp + 1) + ext_ssa_asym * asym_unitless);
  });

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ilev_tropp), [&](int kk) {
    const Real lyr_thk = zi(kk) - zi(kk + 1);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, nswbands), [&](int i) {
      // NOTE: shape of ext_cmip6_sw_inv_m (nswbands,pver)
      const Real ext_unitless = lyr_thk * ext_cmip6_sw_inv_m(i, kk);
      const Real asym_unitless = af_cmip6_sw(kk, i);
      const Real ext_ssa = ext_unitless * ssa_cmip6_sw(kk, i);
      const Real ext_ssa_asym = ext_ssa * asym_unitless;
      tau(i, kk + 1) = ext_unitless;
      tau_w(i, kk + 1) = ext_ssa;
      tau_w_g(i, kk + 1) = ext_ssa_asym;
      tau_w_f(i, kk + 1) = ext_ssa_asym * asym_unitless;
    }); // end nswbands
  });
  // } // kk

} // volcanic_cmip_sw

// similar to compute_odap_volcanic_at_troplayer_lw
// but odap_aer has layout (nlwbands, pver)
KOKKOS_INLINE_FUNCTION
void compute_odap_volcanic_at_troplayer_lw(const ThreadTeam &team,
                                           const int ilev_tropp,
                                           const ConstColumnView &zi,
                                           const View2D &ext_cmip6_lw_inv_m,
                                           const View2D &odap_aer) {
  /* Update odap_aer with a combination read in volcanic aerosol extinction
   [1/m] (50%) and module computed values (50%).

   intent-ins
    ncol
   trop_level(:)
   zi(:,:) geopotential height above surface at interfaces [m]
   ext_cmip6_lw_inv_m(:,:,:) long wave extinction in the units of [1/m]

   intent-inouts
    odap_aer(:,:,:)   [fraction] absorption optical depth, per layer
    [unitless]

   local
   integer :: icol, ilev_tropp
   real(r8) :: lyr_thk layer thickness [m]
   do icol = 1, ncol
   ilev_tropp = trop_level(icol) tropopause level */

  // Need to have these defined outside kernels
  // otherwise undefined errors on cuda dev
  auto _nlwbands = nlwbands;

  const Real lyr_thk =
      zi(ilev_tropp) - zi(ilev_tropp + 1); // compute layer thickness in meters
  constexpr Real half = 0.5;
  // update taus with 50% contributuions from the volcanic input file
  // and 50% from the existing model computed values at the tropopause layer
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, _nlwbands), [&](int i) {
    odap_aer(i, ilev_tropp) =
        half * (odap_aer(i, ilev_tropp) +
                (lyr_thk * ext_cmip6_lw_inv_m(ilev_tropp, i)));
  });

} // compute_odap_volcanic_at_troplayer_lw

// similar to compute_odap_volcanic_above_troplayer_lw
// but odap_aer has layout (nlwbands, pver)
KOKKOS_INLINE_FUNCTION
void compute_odap_volcanic_above_troplayer_lw(const ThreadTeam &team,
                                              const int ilev_tropp,
                                              const ConstColumnView &zi,
                                              const View2D &ext_cmip6_lw_inv_m,
                                              const View2D &odap_aer) {

  /*     Above the tropopause, the read in values from the file include both
       the stratospheric
   and volcanic aerosols. Therefore, we need to zero out odap_aer above the
   tropopause and populate it exclusively from the read in values.

   intent-ins
   integer, intent(in) :: pver, ncol
   integer, intent(in) :: trop_level(:)

   real(r8), intent(in) :: zi(:,:) geopotential height above surface at
   interfaces [m] real(r8), intent(in) :: ext_cmip6_lw_inv_m(:,:,:) long wave
   extinction in the units of [1/m]

   intent-inouts
   real(r8), intent(inout) :: odap_aer(:,:,:)  [fraction] absorption optical
   depth, per layer [unitless]

   local
   integer :: ipver, icol, ilev_tropp
   real(r8) :: lyr_thk layer thickness [m]

   As it will be more efficient for FORTRAN to loop over levels and then
   columns, the following loops are nested keeping that in mind*/

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, ilev_tropp), [&](int kk) {
    const Real lyr_thk =
        zi(kk) - zi(kk + 1); //  compute layer thickness in meters
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, nlwbands), [&](int i) {
      odap_aer(i, kk) = lyr_thk * ext_cmip6_lw_inv_m(kk, i);
    }); // end i
  });   // end kk

} // compute_odap_volcanic_above_troplayer_lw

/* Read the tropopause pressure in from a file containging a climatology. The
   data is interpolated to the current dat of year and latitude.

   NOTE: The data is read in during tropopause_init and stored in the module
   variable trop */

#if 0
  KOKKOS_INLINE_FUNCTION
void tropopause_climate(lchnk,ncol,pmid,pint,temp,zm,zi,    &   in
             tropLev,tropP,tropT,tropZ)   {
// TO be ported...
} // tropopause_climate
#endif
KOKKOS_INLINE_FUNCTION
int tropopause_or_quit(const ConstColumnView &pmid, const ConstColumnView &pint,
                       const ConstColumnView &temperature,
                       const ConstColumnView &zm, const ConstColumnView &zi) {
  /* Find tropopause or quit the simulation if not found

   lchnk             number of chunks
   ncol              number of columns
   pmid(:,:)         midpoint pressure [Pa]
   pint(:,:)         interface pressure [Pa]
   temperature(:,:)  temperature [K]
   zm(:,:)           geopotential height above surface at midpoints [m]
   zi(:,:)           geopotential height above surface at interfaces [m]
   return value [out]
   trop_level(pcols) return value

   trop_level has a value for tropopause for each column
   call tropopause_find(lchnk, ncol, pmid, pint, temperature, zm, zi, & in
          trop_level) out */
  int trop_level = 0;
  tropopause::tropopause_twmo(pmid, pint, temperature, zm, zi, trop_level);

  if (trop_level < -1) {
    Kokkos::abort("aero_rad_props: tropopause not found\n");
  }

  // Need to ported default_backup, i.e., tropopause_climate

  return trop_level;
} // tropopause_or_quit

KOKKOS_INLINE_FUNCTION
void aero_rad_props_sw(const ThreadTeam &team, const Real dt,
                       mam4::Prognostics &progs, const Atmosphere &atm,
                       const ConstColumnView &zi, const ConstColumnView &pdel,
                       const View2D &ssa_cmip6_sw, const View2D &af_cmip6_sw,
                       const View2D &ext_cmip6_sw_m, const View2D &tau,
                       const View2D &tau_w, const View2D &tau_w_g,
                       const View2D &tau_w_f,
                       // FIXME
                       const AerosolOpticsDeviceData &aersol_optics_data,
                       const CalcsizeData &calcsizedata, Real &aodvis,
                       const View1D &work) {

  const ConstColumnView temperature = atm.temperature;
  const ConstColumnView pmid = atm.pressure;
  const ConstColumnView zm = atm.height;
  const ConstColumnView cldn = atm.cloud_fraction;
  const ConstColumnView pint = atm.interface_pressure;
  const ConstColumnView pdeldry = atm.hydrostatic_dp;

  /* call outfld('extinct_sw_inp',ext_cmip6_sw(:,:,idx_sw_diag), pcols, lchnk)

   Return bulk layer tau, omega, g, f for all spectral intervals.

   Arguments
   pmid(:,:)         midpoint pressure [Pa]
   pint(:,:)         interface pressure [Pa]
   temperature(:,:)  temperature [K]
   zm(:,:)           geopotential height above surface at midpoints [m]
   zi(:,:)           geopotential height above surface at interfaces [m]
    state_q(:,:,:)
   pdel(:,:)
   pdeldry(:,:)
   cldn(:,:)
   NOTE: ext_cmip6_sw move unit conversion from km to m outside of this
   function NOTE: ext_cmip6_sw (nswbands, pver) ext_cmip6_sw(:,:,:) [1/m]
   ssa_cmip6_sw(:,:,:)
   af_cmip6_sw(:,:,:)

   nnite                 number of night columns
   idxnite(:)            local column indices of night columns
   is_cmip6_volc         true if cmip6 style volcanic file is read otherwise
   false
   dt                    time step (s)

   qqcw(:)                Cloud borne aerosols mixing ratios [kg/kg or 1/kg]
   tau    (pcols,0:pver,nswbands)  aerosol extinction optical depth
   tau_w  (pcols,0:pver,nswbands)  aerosol single scattering albedo * tau
   tau_w_g(pcols,0:pver,nswbands)  aerosol assymetry parameter * tau * w
   tau_w_f(pcols,0:pver,nswbands)  aerosol forward scattered fraction * tau *
   w

   FORTRAN REFACTOR: This is done to fill invalid values in columns where
   pcols>ncol C++ port can ignore this as C++ model is a single column model
    initialize to conditions that would cause failure
    tau     (:,:,:) = -100._r8
    tau_w   (:,:,:) = -100._r8
    tau_w_g (:,:,:) = -100._r8
    tau_w_f (:,:,:) = -100._r8

    top layer (ilev = 0) has no aerosol (ie tau = 0)
    also initialize rest of layers to accumulate od's
   tau    (1:ncol,:,:) = 0._r8
   tau_w  (1:ncol,:,:) = 0._r8
   tau_w_g(1:ncol,:,:) = 0._r8
   tau_w_f(1:ncol,:,:) = 0._r8

   Find tropopause (or quit simulation if not found) as extinction should be
   applied only above tropopause */
  const int ilev_tropp = tropopause_or_quit(pmid, pint, temperature, zm, zi);
  modal_aero_sw(team, dt, progs, atm, pdel, pdeldry, tau, tau_w, tau_w_g,
                tau_w_f, aersol_optics_data, calcsizedata, aodvis, work);

  team.team_barrier();

  // Update tau, tau_w, tau_w_g, and tau_w_f with the read in values of
  // extinction, ssa and asymmetry factors
  volcanic_cmip_sw(team, zi, ilev_tropp, ext_cmip6_sw_m, ssa_cmip6_sw,
                   af_cmip6_sw, tau, tau_w, tau_w_g, tau_w_f);

  /*  Diagnostic output of total aerosol optical properties
    currently implemented for climate list only
   FIXME: to be ported
   call aer_vis_diag_out(lchnk, ncol, nnite, idxnite, tau(:,:,idx_sw_diag))
  */

} // aero_rad_props_sw

KOKKOS_INLINE_FUNCTION
void aero_rad_props_lw(
    // inputs
    const ThreadTeam &team, const Real dt, mam4::Prognostics &progs,
    const Atmosphere &atm, const ConstColumnView &zi,
    const ConstColumnView &pdel, const View2D &ext_cmip6_lw_m,
    const AerosolOpticsDeviceData &aersol_optics_data,
    const CalcsizeData &calcsizedata,
    // output
    const View2D &odap_aer

) {

  const ConstColumnView temperature = atm.temperature;
  const ConstColumnView pmid = atm.pressure;
  const ConstColumnView zm = atm.height;
  const ConstColumnView cldn = atm.cloud_fraction;
  const ConstColumnView pint = atm.interface_pressure;
  const ConstColumnView pdeldry = atm.hydrostatic_dp;

  /*Purpose: Compute aerosol transmissions needed in absorptivity
   emissivity calculations

   Intent-ins
    is_cmip6_volc     flag for using cmip6 style volc emissions
    dt                time step[s]
    lchnk             number of chunks
    ncol              number of columns
    pmid(:,:)         midpoint pressure [Pa]
    pint(:,:)         interface pressure [Pa]
    temperature(:,:)  temperature [K]
    zm(:,:)           geopotential height above surface at midpoints [m]
    zi(:,:)           geopotential height above surface at interfaces [m]
    state_q(:,:,:)
    pdel(:,:)
    pdeldry(:,:)
    cldn(:,:)
    ext_cmip6_lw_m(:,:,:) long wave extinction in the units of [1/m]
    qqcw(:)    Cloud borne aerosols mixing ratios [kg/kg or 1/kg]

   intent-outs
   Note: using emaxx layout
    odap_aer(pcols,nlwbands, pver)  [fraction] absorption optical depth, per
    layer [unitless]
   Compute contributions from the modal aerosols.*/
  modal_aero_lw(team, dt, progs, atm, pdel, pdeldry, aersol_optics_data,
                calcsizedata,
                // outputs
                odap_aer);

  /* FIXME: port tropopause_or_quit
   Find tropopause or quit simulation if not found
   trop_level(1:pcols) = tropopause_or_quit(lchnk, ncol, pmid, pint,
   temperature, zm, zi)*/
  const int ilev_tropp = tropopause_or_quit(pmid, pint, temperature, zm, zi);
  team.team_barrier();

  // /*We are here because tropopause is found, update taus with 50%
  // contributuions from the volcanic input file and 50% from the existing model
  // computed values at the tropopause layer
  compute_odap_volcanic_at_troplayer_lw(team, ilev_tropp, zi, ext_cmip6_lw_m,
                                        odap_aer);
  // /* Above the tropopause, the read in values from the file include both the
  //  stratospheric
  //   and volcanic aerosols. Therefore, we need to zero out odap_aer above the
  //   tropopause and populate it exclusively from the read in values.
  compute_odap_volcanic_above_troplayer_lw(team, ilev_tropp, zi, ext_cmip6_lw_m,
                                           odap_aer);
  // call outfld('extinct_lw_bnd7',odap_aer(:,:,idx_lw_diag), pcols, lchnk)

} // aero_rad_props_lw

} // namespace aero_rad_props
} // end namespace mam4

#endif
