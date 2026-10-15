#ifndef MAM4XX_AERO_MODEL_EMISSIONS_HPP
#define MAM4XX_AERO_MODEL_EMISSIONS_HPP

#include "mam4_constants.hpp"
#include "mam4_math.hpp"
#include "mam4_types.hpp"

namespace mam4::aero_model_emissions {

using const_view_1d = DeviceType::view_1d<const Real>;
using const_view_2d = DeviceType::view_2d<const Real>;

using view_1d = DeviceType::view_1d<Real>;

// essentially everything in this namespace falls in the BAD CONSTANT category
// ... thus, the name
namespace BAD_CONSTANTS {

constexpr int pcnst = mam4::pcnst;

enum class FluxType { MassFlux, NumberFlux };

// =============================================================================
//      Dust Parameters/Constants
// =============================================================================
// nbin corresponds to dst_a1, dst_a3
constexpr int dust_nbin = 2;
// nnum corresponds to num_a1, num_a3
constexpr int dust_nnum = 2;
// number of entries in the dust flux input array
constexpr int dust_nflux_in = 4;
// Aerosol density [kg m-3]
constexpr Real dust_density = 2.5e3;

struct DustEmissionsData {
  // Kok11: fractions of bin (0.1-1) and bin (1-10) in size 0.1-10
  KOKKOS_INLINE_FUNCTION Real dust_emis_mass_frac_mode(const int index) const {
    static constexpr Real dust_emis_mass_frac_mode_[dust_nbin] = {0.011, 0.989};
    return dust_emis_mass_frac_mode_[index];
  }
  // got this value from fortran mam4::dust_model.F90:26
  KOKKOS_INLINE_FUNCTION Real dust_dmt_grd(const int index) const {
    static constexpr Real dust_dmt_grd_[dust_nbin + 1] = {1.0e-7, 1.0e-6,
                                                          1.0e-5};
    return dust_dmt_grd_[index];
  }
  // tuning parameter for dust emissions
  Real dust_emis_scale_factor = 1.0;
  // this is set by init_dust_dmt_vwr() which takes in a constant value for
  // dust_dmt_grd (above)
  Real dust_dmt_vwr[dust_nbin];
};
// =============================================================================
//      Sea Salt Parameters/Constants
// =============================================================================
constexpr int salt_nsection = 31;
// TODO: Unnecessary indices?
// NOTE: to keep with standard c++ looping convention
//       e.g., for i in [beg, end), we subtract 1 from the fortran begin
//       indices, but not the end
constexpr int sec1_beg = 1 - 1;
constexpr int sec1_end = 9;
constexpr int sec2_beg = 10 - 1;
constexpr int sec2_end = 13;
constexpr int sec3_beg = 14 - 1;
constexpr int sec3_end = 21;
constexpr int sec4_beg = 22 - 1;
constexpr int sec4_end = salt_nsection;
// Aerosol density [kg m-3]
constexpr Real seasalt_density = 2.2e3;

// NOTE: nsalt is a model-fixed value for our purposes
// (mam4 => ntot_amode = 4 => ntot_amode = 3), so it seems like
// needless pain to actually do the max calculation and not make this a
// constexpr so we can use it and others as size arguments up here
// NOTE: in the fortran, nsalt is used as an offset to jump over the first 3
// entries in the seasalt_indices array. as such, we may need to subtract 1 when
// using as an index
// const int nsalt = mam4::max(3, ntot_amode - 3);
constexpr int nsalt = 3;
constexpr int nnum = nsalt;
constexpr int nsalt_om = 3;
constexpr int nnum_om = 1;
constexpr int om_num_modes = 3;

constexpr int seasalt_nbin = nsalt + nsalt_om;
constexpr int seasalt_nnum = nnum + nnum_om;

// =============================================================================
//      Marine Organic Matter Parameters/Constants
// =============================================================================
constexpr int n_ocean_data = 4;
constexpr int n_organic_species = 3;
// FIXME: probably don't need this?
constexpr int n_organic_species_max = 3;
// is this different from n_organic_species? both appear in the fortran
constexpr int organic_num_modes = 3;
// Parameters for organic sea salt emissions
// approx volume density of salt in seawater [g/m^3]
constexpr Real vol_density_NaCl_seawater = 35875.0;
// Namelist variables for parameterization specification
// Bubble film thickness [m]
// NOTE: this is given as 0.1e-6 in the fortran
constexpr Real l_bub = 1.0e-7;
// smallest ocean organic concentration allowed
constexpr Real small_oceanorg = 1.0e-30;

struct SeasaltEmissionsData {
  // use Ekmam's ss
  // radius [m]
  Real rdry[salt_nsection];
  // multiply rm with 1.814 because it should be RH=80% and not dry particles
  // for the parameterization
  // [um]
  Real rm[salt_nsection];
  // use in [Manahan]
  Real bm[salt_nsection];

  // constants for calculating emission polynomial
  Real consta[salt_nsection], constb[salt_nsection];

  // FIXME: BAD CONSTANTS
  // only use up to ~20um
  KOKKOS_INLINE_FUNCTION Real Dg(const int index) const {
    static constexpr Real Dg_[salt_nsection] = {
        2.0e-8,    2.5e-8,    3.2e-8,   4.0e-8,   5.1e-8,   6.5e-8,   8.2e-8,
        1.04e-7,   1.32e-7,   1.67e-7,  2.11e-7,  2.67e-7,  3.38e-7,  4.28e-7,
        5.41e-7,   6.85e-7,   8.67e-7,  1.098e-6, 1.389e-6, 1.759e-6, 2.226e-6,
        2.818e-6,  3.571e-6,  4.526e-6, 5.735e-6, 7.267e-6, 9.208e-6, 1.1668e-5,
        1.4786e-5, 1.8736e-5, 2.3742e-5};
    return Dg_[index];
  }

  // OM:OC mass ratios for input fields (mpoly, mprot, mlip)
  // Order: mpoly, mprot, mlip
  // FIXME: BAD CONSTANTS
  KOKKOS_INLINE_FUNCTION Real OM_to_OC_in(const int index) const {
    static constexpr Real OM_to_OC_in_[n_organic_species] = {2.3, 2.2, 1.3};
    return OM_to_OC_in_[index];
  }
  // Langmuir parameters (inverse C_1/2)  [m3 mol-1]
  KOKKOS_INLINE_FUNCTION Real alpha_org(const int index) const {
    static constexpr Real alpha_org_[n_organic_species] = {90.58, 25175.0,
                                                           18205.0};
    return alpha_org_[index];
  }
  // Molecular weights [g mol-1]
  KOKKOS_INLINE_FUNCTION Real mw_org(const int index) const {
    static constexpr Real mw_org_[n_organic_species] = {250000.0, 66463.0,
                                                        284.0};
    return mw_org_[index];
  }

  KOKKOS_INLINE_FUNCTION Real dens_srf_org(const int index) const {
    // mass per sq. m at saturation [Mw_org / a_org]
    static constexpr Real dens_srf_org_[n_organic_species] = {0.1376, 0.00219,
                                                              0.002593};
    return dens_srf_org_[index];
  }

  // entries correspond to:
  //        accum, aitken, coarse
  //        accum, aitken, POM accum
  // the indices here are meant to correspond to the seasalt "bins"
  KOKKOS_INLINE_FUNCTION Real seasalt_size_range_lo(const int index) const {
    static constexpr Real seasalt_size_range_lo_[seasalt_nbin] = {
        8.0e-8, 2.0e-8, 1.0e-6, 8.0e-8, 2.0e-8, 8.0e-8};
    return seasalt_size_range_lo_[index];
  }

  KOKKOS_INLINE_FUNCTION Real seasalt_size_range_hi(const int index) const {
    static constexpr Real seasalt_size_range_hi_[seasalt_nbin] = {
        1.0e-6, 8.0e-8, 1.0e-5, 1.0e-6, 8.0e-8, 1.0e-6};
    return seasalt_size_range_hi_[index];
  }

  KOKKOS_INLINE_FUNCTION int seasalt_indices(const int index) const {
    static constexpr int seasalt_indices_[seasalt_nbin + seasalt_nnum] = {
        20, 25, 29, 21, 26, 38, 22, 27, 35, 39};
    return seasalt_indices_[index];
  }

  KOKKOS_INLINE_FUNCTION int organic_num_idx(const int index) const {
    static constexpr int organic_num_idx_[organic_num_modes] = {0, 1, 3};
    return organic_num_idx_[index];
  }

  Real mpoly, mprot, mlip;

  // tuning parameter for seasalt emissions
  Real seasalt_emis_scale_factor = 1.0;
};

} // namespace BAD_CONSTANTS

using namespace BAD_CONSTANTS;

struct OnlineEmissionsData {
  Real dust_flux_in[dust_nflux_in] = {};
  Real surface_temp = 0;
  Real u_bottom = 0;
  Real v_bottom = 0;
  Real z_bottom = 0;
  Real ocean_frac = 0;
  Real soil_erodibility = 0;
};

// =============================================================================

// NOTE: this corresponds to dust_common.F90::dust_set_params(), which
// calculates another variable and employs a lot of "bad constants"
KOKKOS_INLINE_FUNCTION
void init_dust_dmt_vwr(DustEmissionsData &data) {

  // FIXME: BAD CONSTANT
  constexpr int sz_nbr = 200;

  // Introducing particle diameter. Needed by atm model and by dry dep model.
  // Taken from Charlie Zender's subroutines dst_psd_ini, dst_sz_rsl,
  // grd_mk (dstpsd.F90) and subroutine lgn_evl (psdlgn.F90)

  // Charlie allows logarithmic or linear option for size distribution.
  // however, he hardwires the distribution to logarithmic in his code.
  // therefore, I take his logarithmic code only
  // furthermore, if dst_nbr == 4, he overrides the automatic grid
  // calculation
  // he currently works with dst_nbr = 4, so I only take the relevant code
  // if dust_number ever becomes different from 4,
  // must add call grd_mk (dstpsd.F90) as done in subroutine dst_psd_ini
  // note that here dust_number = dst_nbr

  // Max diameter in bin [m]
  Real dmt_min[dust_nbin];
  // Min diameter in bin [m]
  Real dmt_max[dust_nbin];

  // Override automatic grid with preset grid if available
  for (int n = 0; n < dust_nbin; ++n) {
    dmt_min[n] = data.dust_dmt_grd(n);
    dmt_max[n] = data.dust_dmt_grd(n + 1);
  }

  // set dust_dmt_vwr ....
  for (int i = 0; i < dust_nbin; ++i) {
    data.dust_dmt_vwr[i] = 0.0;
  }

  // Bin physical properties
  // [frc] Geometric std dev PaG77 p. 2080 Table1
  constexpr Real gsd_anl = 2.0;
  const Real ln_gsd = mam4::log(gsd_anl);

  // Set a fundamental statistic for each bin
  // Mass median diameter analytic She84 p.75 Table 1 [m]
  // FIXME: FIXME: this appears to be a bug--vma is set, and then re-set
  //        maybe the second one should be dmt_nma?
  // // FIXME: BAD CONSTANTS
  Real dmt_vma = 2.524e-6;
  dmt_vma = 3.5e-6;
  // Compute analytic size statistics
  // Convert mass median diameter to number median particle diameter [m]
  Real dmt_nma = dmt_vma * mam4::exp(-3.0 * square(ln_gsd));

  // [m] Size Bin minima
  Real sz_min[sz_nbr];
  // [m] Size Bin maxima
  Real sz_max[sz_nbr];
  // [m] Size Bin centers
  Real sz_ctr[sz_nbr];
  // [m] Size Bin widths
  Real sz_dlt[sz_nbr];

  // [m3 m-3] Volume concentration resolved
  Real vlm_rsl[dust_nbin] = {0.0};

  // Compute resolved size statistics for each size distribution
  // In C. Zender's code call dst_sz_rsl
  for (int n = 0; n < dust_nbin; ++n) {
    // Factor for logarithmic grid
    Real series_ratio = mam4::pow((dmt_max[n] / dmt_min[n]), (1.0 / sz_nbr));
    sz_min[0] = dmt_min[n];
    // NOTE: Loop starts at 1 (2 in fortran code)
    for (int m = 1; m < sz_nbr; ++m) {
      sz_min[m] = sz_min[m - 1] * series_ratio;
    }
    // Derived grid values
    // NOTE: Loop ends at sz_nbr - 2 (fortran: sz_nbr - 1)
    for (int m = 0; m < sz_nbr - 1; ++m) {
      sz_max[m] = sz_min[m + 1];
    }
    // set the final entry
    sz_max[sz_nbr - 1] = dmt_max[n];
    // Final derived grid values
    for (int m = 0; m < sz_nbr; ++m) {
      sz_ctr[m] = 0.5 * (sz_min[m] + sz_max[m]);
      sz_dlt[m] = sz_max[m] - sz_min[m];
    }

    constexpr Real pi = Constants::pi;
    // Lognormal distribution at sz_ctr
    // Factor in lognormal distribution
    // NOTE: original variable name in fortran: lngsdsqrttwopi_rcp
    const Real lnN_factor = 1.0 / (ln_gsd * mam4::sqrt(2.0 * pi));

    for (int m = 0; m < sz_nbr; ++m) {
      // Evaluate lognormal distribution for these sizes (call lgn_evl)
      Real tmp = mam4::log(sz_ctr[m] / dmt_nma) / ln_gsd;
      Real lgn_dst = lnN_factor * mam4::exp(-0.5 * square(tmp)) / sz_ctr[m];
      Real coeff = pi / 6.0 * mam4::pow(sz_ctr[m], 3) * lgn_dst * sz_dlt[m];
      // Integrate moments of size distribution
      data.dust_dmt_vwr[n] += sz_ctr[m] * coeff;
      vlm_rsl[n] += coeff;
    } // end for (m)
    // Mass weighted diameter resolved [m]
    data.dust_dmt_vwr[n] /= vlm_rsl[n];
  } // end for (dust_nbin)
} // end init_dust_dmt_vwr()

KOKKOS_INLINE_FUNCTION
void dust_emis(
    // in
    const Real &dust_density, const Real (&dust_flux_in)[dust_nflux_in],
    const DustEmissionsData &data, const Real &soil_erodibility,
    //  inout
    Real (&cflux)[pcnst]) {
  // dust_flux_in: dust emission fluxes in
  // cflux: emission fluxes in MAM modes [{kg,#}/m^2/s]
  // soil_erod: soil erodibility factor [unitless]

  // FIXME: BAD CONSTANT
  constexpr Real soil_erod_threshold = 0.1;
  // FIXME: BAD CONSTANT
  constexpr int dust_indices[dust_nbin + dust_nnum] = {19, 28, 22, 35};

  // dust_density[dust_nbin] -- Mass weighted diameter resolved [m]
  Real dust_mass_to_num[dust_nbin];

  for (int ibin = 0; ibin < dust_nbin; ++ibin) {
    dust_mass_to_num[ibin] = 6.0 / (Constants::pi * dust_density *
                                    mam4::pow(data.dust_dmt_vwr[ibin], 3));
  }

  if (soil_erodibility >= soil_erod_threshold) {
    // in fortran, done inside loop as: sum( -dust_flux_in(:) )
    Real dust_flux_neg_sum = 0.0;
    for (int ibin = 0; ibin < dust_nflux_in; ++ibin) {
      dust_flux_neg_sum -= dust_flux_in[ibin];
    }
    // rebin and adjust dust emissions..
    // Correct the dust input flux calculated by CLM, which uses size
    // distribution in Zender03 to calculate fraction of bin (0.1-10um) in range
    // (0.1-20um) = 0.87 based on Kok11, that fraction is 0.73
    // FIXME: BAD CONSTANTS
    constexpr Real frac_ratio = 0.73 / 0.87;
    for (int ibin = 0; ibin < dust_nbin; ++ibin) {
      const int idx_dust = dust_indices[ibin];
      // FIXME: BAD CONSTANT
      cflux[idx_dust] = dust_flux_neg_sum * frac_ratio *
                        data.dust_emis_mass_frac_mode(ibin) * soil_erodibility /
                        data.dust_emis_scale_factor * 1.15;
      const int inum = dust_indices[ibin + dust_nbin];
      cflux[inum] = cflux[idx_dust] * dust_mass_to_num[ibin];
    }
  } else {
    for (int ibin = 0; ibin < dust_nbin; ++ibin) {
      const int idx_dust = dust_indices[ibin];
      const int inum = dust_indices[ibin + dust_nbin];
      cflux[idx_dust] = 0.0;
      cflux[inum] = 0.0;
    }
  }
  // // local version (unnecessary?) that's zero if less than the threshold
  // // NOTE: if this ends up true, then it results in
  // //       soil_erod = 0 and cflux[:] = 0, making all of the below pointless
  // Real soil_erod =
  //     soil_erodibility < soil_erod_threshold ? 0.0 : soil_erodibility;

} // end dust_emis()

KOKKOS_INLINE_FUNCTION
void init_seasalt_sections(SeasaltEmissionsData &data /* inout */) {
  // use Ekman's ssd
  // multiply rm with 1.814 because it should be RH=80% and not dry particles
  // for the parameterization
  // FIXME: BAD CONSTANTS
  for (int i = 0; i < salt_nsection; ++i) {
    data.rdry[i] = data.Dg(i) / 2.0;
    data.rm[i] = 1.814 * data.rdry[i] * 1.0e6;
    data.bm[i] = (0.380 - mam4::log10(data.rm[i])) / 0.65;
  }

  // FIXME: BAD CONSTANTS
  // calculate constants from emission polynomials
  for (int isec = sec1_beg; isec < sec1_end; ++isec) {
    data.consta[isec] = -2.576e35 * mam4::pow(data.Dg(isec), 4) +
                        5.932e28 * mam4::pow(data.Dg(isec), 3) -
                        2.867e21 * mam4::pow(data.Dg(isec), 2) -
                        3.003e13 * data.Dg(isec) - 2.881e6;
    data.constb[isec] = 7.188e37 * mam4::pow(data.Dg(isec), 4) -
                        1.616e31 * mam4::pow(data.Dg(isec), 3) +
                        6.791e23 * mam4::pow(data.Dg(isec), 2) +
                        1.829e16 * data.Dg(isec) + 7.609e8;
  }
  for (int isec = sec2_beg; isec < sec2_end; ++isec) {
    data.consta[isec] = -2.452e33 * mam4::pow(data.Dg(isec), 4) +
                        2.404e27 * mam4::pow(data.Dg(isec), 3) -
                        8.148e20 * mam4::pow(data.Dg(isec), 2) +
                        1.183e14 * data.Dg(isec) - 6.743e6;
    data.constb[isec] = 7.368e35 * mam4::pow(data.Dg(isec), 4) -
                        7.310e29 * mam4::pow(data.Dg(isec), 3) +
                        2.528e23 * mam4::pow(data.Dg(isec), 2) -
                        3.787e16 * data.Dg(isec) + 2.279e9;
  }
  for (int isec = sec3_beg; isec < sec3_end; ++isec) {
    data.consta[isec] = 1.085e29 * mam4::pow(data.Dg(isec), 4) -
                        9.841e23 * mam4::pow(data.Dg(isec), 3) +
                        3.132e18 * mam4::pow(data.Dg(isec), 2) -
                        4.165e12 * data.Dg(isec) + 2.181e6;
    data.constb[isec] = -2.859e31 * mam4::pow(data.Dg(isec), 4) +
                        2.601e26 * mam4::pow(data.Dg(isec), 3) -
                        8.297e20 * mam4::pow(data.Dg(isec), 2) +
                        1.105e15 * data.Dg(isec) - 5.800e8;
  }
  for (int isec = sec4_beg; isec < sec4_end; ++isec) {
    // use monahan
    data.consta[isec] =
        (1.373 * mam4::pow(data.rm[isec], -3) *
         (1 + 0.057 * mam4::pow(data.rm[isec], 1.05)) *
         mam4::pow(10, 1.19 * mam4::exp(-square(data.bm[isec])))) *
        (data.rm[isec] - data.rm[isec - 1]);
  }
} // end init_seasalt_sections

// NOTE: this function is kind of pointless because our model is hard-coded.
// as such, I've matched up seasalt names/indices above in a hardcoded array
KOKKOS_INLINE_FUNCTION
void init_seasalt(SeasaltEmissionsData &data /* inout */) {
  init_seasalt_sections(data);
} // end init_seasalt()

KOKKOS_INLINE_FUNCTION
void calc_seasalt_fluxes(
    // in
    const Real &surface_temp, const Real &u10cubed,
    const Real (&consta)[salt_nsection], const Real (&constb)[salt_nsection],
    // out
    Real (&fluxes)[salt_nsection]) {
  // Calculations of source strength and size distribution
  // NB the 0.1 is the dlogDp we have to multiply with to get the flux, but the
  // value depends of course on what dlogDp you have.
  // You will also have to change the sections of Dg if you use a different
  // number of size bins with different intervals.

  // whitecap area
  // FIXME: BAD CONSTANTS
  const Real wc_area = 3.84e-6 * u10cubed * 0.1;

  // calculate number flux fluxes (#/m2/s)
  for (int isec = sec1_beg; isec < sec1_end; ++isec) {
    fluxes[isec] = wc_area * (surface_temp * consta[isec] + constb[isec]);
  }
  for (int isec = sec2_beg; isec < sec2_end; ++isec) {
    fluxes[isec] = wc_area * (surface_temp * consta[isec] + constb[isec]);
  }
  for (int isec = sec3_beg; isec < sec3_end; ++isec) {
    fluxes[isec] = wc_area * (surface_temp * consta[isec] + constb[isec]);
  }
  for (int isec = sec4_beg; isec < sec4_end; ++isec) {
    fluxes[isec] = consta[isec] * u10cubed;
  }
} // end calc_seasalt_fluxes()

KOKKOS_INLINE_FUNCTION
void calculate_seasalt_numflux_in_bins(
    // in
    const Real &surface_temp, const Real &u_bottom, const Real &v_bottom,
    const Real &z_bottom, const Real (&consta)[salt_nsection],
    const Real (&constb)[salt_nsection],
    //  out
    Real (&fi)[salt_nsection]) {

  // surface_temp: sea surface temperature [K]
  // fi: sea salt number fluxes in each size bin [#/m2/s]

  // m roughness length over oceans--from ocean model
  // FIXME: BAD CONSTANT
  constexpr Real z0 = 0.0001;

  // Needed in Gantt et al. calculation of organic mass fraction
  Real u10 = mam4::sqrt(square(u_bottom) + square(v_bottom));

  // move the winds to 10m high from the midpoint of the gridbox:
  // follows Tie and Seinfeld and Pandis, p.859 with math.
  // u10cubed is defined as the 3.41 power of 10m wind
  // FIXME: terrible variable name
  // FIXME: BAD CONSTANT
  Real u10cubed = u10 * mam4::log(10.0 / z0) / mam4::log(z_bottom / z0);

  // we need them to the 3.41 power, according to Gong et al., 1997
  // FIXME: BAD CONSTANT
  u10cubed = mam4::pow(u10cubed, 3.41);

  calc_seasalt_fluxes(
      // in
      surface_temp, u10cubed, consta, constb,
      // out
      fi);
} // end calculate_seasalt_numflux_in_bins()

KOKKOS_INLINE_FUNCTION
void seasalt_emis_flux_calc(
    // in
    const Real (&fi)[salt_nsection], const Real &ocean_frac,
    const FluxType &flux_type, const SeasaltEmissionsData &data,
    // inout
    Real (&cflux)[pcnst]) {

  // input
  // fi: sea salt number fluxes in each size bin [#/m2/s]
  // ocean_frac: ocean fraction [unitless]
  // flux_type: flux to be calculated--number flux == 0 or mass flux == 1
  //            NOTE: changed to an enum class for clarity

  // output
  // cflux: mass and number emission fluxes for aerosols [{kg,#}/m2/s]

  const int num_idx_append =
      (flux_type == FluxType::NumberFlux) ? nsalt + nsalt_om : 0;

  for (int ispec = 0; ispec < nsalt; ++ispec) {
    int mode_idx = data.seasalt_indices(num_idx_append + ispec);
    if (mode_idx > 0) {
      if (flux_type == FluxType::MassFlux) {
        // FIXME (from fortran team):
        // If computing number flux, zero out the array
        // Note for C++ port, ideally initialization of clfx should be done
        // for both number and mass modes. We only zero-out number, because
        // zero-out mass fluxes cause NBFB
        cflux[mode_idx] = 0.0;
      }
      for (int ibin = 0; ibin < salt_nsection; ++ibin) {
        Real cflux_tmp = 0.0;
        if (data.Dg(ibin) >= data.seasalt_size_range_lo(ispec) &&
            data.Dg(ibin) < data.seasalt_size_range_hi(ispec)) {
          cflux_tmp = fi[ibin] * ocean_frac * data.seasalt_emis_scale_factor;

          // For mass fluxes, multiply by the diameter
          // Note for C++ port, 4.0/3.0*pi*rdry[ibin]**3*seasalt_density
          // can be factored out in a function, but it is not done here
          // as the results might not stay BFB
          if (flux_type == FluxType::MassFlux) {
            cflux_tmp *= 4.0 / 3.0 * Constants::pi *
                         mam4::pow(data.rdry[ibin], 3) * seasalt_density;
          }
          // Mixing state 3: internal mixture, add OM to mass and number
          cflux[mode_idx] += cflux_tmp;
        } // end if (Dg(ibin))
      }   // end for (ibin)
    }     // end if (mode_idx)
  }       // end for (ispec)
} // end seasalt_emis_flux_calc()

KOKKOS_INLINE_FUNCTION
void seasalt_emis(
    // in
    const Real (&fi)[salt_nsection], const Real &ocean_frac,
    const SeasaltEmissionsData &data,
    // inout
    Real (&cflux)[pcnst]) {
  // calculate seasalt number emission fluxes
  seasalt_emis_flux_calc(
      // in
      fi, ocean_frac, FluxType::NumberFlux, data,
      // inout
      cflux);
  // calculate seasalt mass emission fluxes
  seasalt_emis_flux_calc(
      // in
      fi, ocean_frac, FluxType::MassFlux, data,
      // inout
      cflux);
} // end seasalt_emis()

KOKKOS_INLINE_FUNCTION
void om_fraction_accum_aitken(
    // in
    Real &om_seasalt_in, const SeasaltEmissionsData &data,
    // out
    Real (&om_seasalt)[salt_nsection]) {
  // -----------------------------------------------------------------------
  //  Purpose:
  //  Put OM fraction directly into aitken and accumulation modes and don't
  //  exceed om_seasalt_max.

  // om_seasalt_in: mass fraction [unitless]
  // om_seasalt: mass fraction for each size bin [unitless]
  // -----------------------------------------------------------------------

  // FIXME: BAD CONSTANT
  constexpr Real om_seasalt_max = 1.0;

  // Initialize array and set to zero for "fine sea salt" and
  // "coarse sea salt" modes
  for (int ibin = 0; ibin < salt_nsection; ++ibin) {
    om_seasalt[ibin] = 0.0;
  }

  // distribute OM fraction!
  for (int ibin = 0; ibin < salt_nsection; ++ibin) {
    // update only in Aitken and accumulation modes
    om_seasalt[ibin] = (data.Dg(ibin) >= data.seasalt_size_range_lo(1)) &&
                               (data.Dg(ibin) < data.seasalt_size_range_hi(0))
                           ? om_seasalt_in
                           : om_seasalt[ibin];
  }

  // For safety, force fraction to be within bounds [0, 1]
  for (int ibin = 0; ibin < salt_nsection; ++ibin) {
    // FIXME: this looks a little odd to me to set x = 0 if (x < 1e-30),
    // especially since the comments states the goal as bounding in [0, 1]
    om_seasalt[ibin] =
        om_seasalt[ibin] > om_seasalt_max ? om_seasalt_max : om_seasalt[ibin];
    om_seasalt[ibin] =
        om_seasalt[ibin] < small_oceanorg ? 0.0 : om_seasalt[ibin];
  }
} // end subroutine om_fraction_accum_aitken()

KOKKOS_INLINE_FUNCTION
void calc_org_matter_seasalt(
    // in
    const SeasaltEmissionsData &data,
    // out
    Real (&mass_frac_bub_section)[n_organic_species_max][salt_nsection],
    Real (&om_seasalt)[salt_nsection]) {
  // Input variables:
  // for Burrows et al. (2014) organic mass fraction
  // data.mpoly
  // data.mprot
  // data.mlip

  // Output variables
  // mass_frac_bub_section: mass fraction per organic class per size bin
  // om_seasalt: mass fraction per size bin [unitless]

  // OMF maximum and minimum values -- max from Rinaldi et al. (2013)
  // FIXME: BAD CONSTANT
  constexpr Real omfrac_max = 0.78;
  constexpr Real liter_to_m3 = 1.0e-3;

  // convert from [kg/mol] to [g/mol]
  constexpr Real mw_carbon = 1.0e3 * Constants::molec_weight_c;

  // fractional surface coverage [unitless]
  Real theta[n_organic_species] = {0.0};
  Real theta_help[n_organic_species] = {0.0};
  // this is not initilized in fortran, but should(?) be ok?
  Real alpha_help = 0.0;
  Real mass_frac_bub_tot = 0.0;
  Real mass_frac_bub[n_organic_species] = {0.0};
  Real mass_frac_bub_help[n_organic_species] = {0.0};
  // mole concentration of organic [mol/m^3]
  Real om_conc[n_organic_species] = {0.0};

  // Convert input fields from [(mol C) L-1] to [(g OM) m-3] and store in single
  // array
  om_conc[0] = data.mpoly * liter_to_m3 * data.OM_to_OC_in(0) * mw_carbon;
  om_conc[1] = data.mprot * liter_to_m3 * data.OM_to_OC_in(1) * mw_carbon;
  om_conc[2] = data.mlip * liter_to_m3 * data.OM_to_OC_in(2) * mw_carbon;

  // Calculate the surface coverage by class
  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    // Bulk mass concentration [mol m-3] = [g m-3] / [g mol-1]
    om_conc[iorg] = om_conc[iorg] / data.mw_org(iorg);
    // use theta_help as work array -- theta_help = alpha(i) * x(i)
    theta_help[iorg] = data.alpha_org(iorg) * om_conc[iorg];
  }
  // Burrows et al. (2014), Eq. 2: all organic classes compete through the
  // dimensionless sum of their concentration-weighted Langmuir terms.
  for (int i = 0; i < n_organic_species; ++i) {
    alpha_help += theta_help[i];
  }

  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    // complete calculation: theta = alpha(i) * x(i)
    //                               / (1 + sum(alpha(i) * x(i)))
    theta[iorg] = theta_help[iorg] / (1.0 + alpha_help);
    // Calculate the organic mass per area (by class) [g m-2]
    // (use mass_frac_bub_help as local work array--
    // organic mass per area in g per m2)
    mass_frac_bub_help[iorg] = theta[iorg] * data.dens_srf_org(iorg);
  }

  // Calculate g NaCl per m2
  // (Redundant, but allows for easier adjustment to l_bub)
  Real dens_srf_NaCl_in_bubsrf = vol_density_NaCl_seawater * l_bub;
  Real sum_mass_frac_bub_help = 0.0;
  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    sum_mass_frac_bub_help += mass_frac_bub_help[iorg];
  }

  // mass_frac_bub = 2 * [g OM m-2] / (2 * [g OM m-2] * [g NaCl m-2])
  // Factor 2 for bubble bilayer (coated on both surfaces of film)
  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    mass_frac_bub[iorg] =
        2.0 * mass_frac_bub_help[iorg] /
        (2.0 * sum_mass_frac_bub_help + dens_srf_NaCl_in_bubsrf);
  }
  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    mass_frac_bub_tot += mass_frac_bub[iorg];
  }
  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    mass_frac_bub[iorg] =
        mass_frac_bub_tot > omfrac_max
            ? mass_frac_bub[iorg] / mass_frac_bub_tot * omfrac_max
            : mass_frac_bub[iorg];
    // Must exceed threshold value small_oceanorg
    mass_frac_bub[iorg] =
        mass_frac_bub[iorg] < small_oceanorg ? 0.0 : mass_frac_bub[iorg];
  }
  mass_frac_bub_tot = 0.0;
  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    mass_frac_bub_tot += mass_frac_bub[iorg];
  }

  // Distribute mass fraction evenly into Aitken and accumulation modes
  om_fraction_accum_aitken(
      // in
      mass_frac_bub_tot, data,
      // out
      om_seasalt);

  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    // NOTE: slicing along iorg makes mass_frac_bub a scalar
    // and mass_frac_bub_section an array with extent salt_nsection
    om_fraction_accum_aitken(
        // in
        mass_frac_bub[iorg], data,
        // out
        mass_frac_bub_section[iorg]);
  }
} // end calc_org_matter_seasalt()

KOKKOS_INLINE_FUNCTION
void calc_marine_organic_numflux(
    // in
    const Real (&fi)[salt_nsection], const Real &ocean_frac,
    const Real (&om_seasalt)[salt_nsection],
    const bool (&emit_this_mode)[organic_num_modes],
    const SeasaltEmissionsData &data,
    //  inout
    Real (&cflux)[pcnst]) {
  // ocean_frac: ocean fraction [unitless]
  // om_seasalt: marine organic aerosol fraction per size bin [unitless]
  // emit_this_mode: logical flags turn on/off marine organic emission in
  //                 aerosol modes

  // output
  // cflux: mass and number emission fluxes for aerosols [kg/m2/s or #/m2/s]

  // this is hardcoded for the model, so I don't think we need this check
  // if (size(organic_num_idx) .eq. 1) then
  //   call endrun( "Error: organic_num_idx is a scalar, but attempting to
  //   calculate MOM. Something bad happened. We should never get here!")
  // endif

  // Loop over OM modes
  for (int ispec = 0; ispec < organic_num_modes; ++ispec) {
    // modes in which to emit OM
    int om_num_idx = data.organic_num_idx(ispec);
    int num_mode_idx = data.seasalt_indices(nsalt + nsalt_om + om_num_idx);
    // add number tracers for organics-only modes
    if (emit_this_mode[ispec]) {
      for (int ibin = 0; ibin < salt_nsection; ++ibin) {
        Real cflux_tmp = 0.0;
        if ((data.Dg(ibin) >= data.seasalt_size_range_lo(nsalt + ispec)) &&
            (data.Dg(ibin) < data.seasalt_size_range_hi(nsalt + ispec))) {
          cflux_tmp = fi[ibin] * ocean_frac * data.seasalt_emis_scale_factor;
          // Mixing state 3: internal mixture, add OM to mass and number
          cflux[num_mode_idx] +=
              cflux_tmp * (1.0 / (1.0 - om_seasalt[ibin]) - 1.0);
        } // end if (Dg)
      }   // end for (ibin)
    }     // end if (mode_idx)
  }       // end for (ispec)
} // end calc_marine_organic_numflux()

KOKKOS_INLINE_FUNCTION
void calc_marine_organic_massflux(
    // in
    const Real (&fi)[salt_nsection], const Real &ocean_frac,
    const Real (&om_seasalt)[salt_nsection],
    const Real (&mass_frac_bub_section)[n_organic_species_max][salt_nsection],
    const bool (&emit_this_mode)[organic_num_modes],
    const SeasaltEmissionsData &data,
    // out
    Real (&cflux)[pcnst]) {

  // ocean_frac: ocean fraction [unitless]
  // om_seasalt: marine organic aerosol fraction per size bin [unitless]
  // mass_frac_bub_section: marine organic aerosol fraction per organic species
  //                        per size bin [unitless]
  // emit_this_mode: logical flags turn on/off marine organic emission in
  //                 aerosol modes

  // output
  // cflux: mass and number emission fluxes for aerosols [kg/m2/s or #/m2/s]

  int mass_mode_idx;
  Real cflux_tmp;
  for (int ispec = 0; ispec < nsalt_om; ++ispec) {
    int idx_salt_offset = nsalt + ispec;
    mass_mode_idx = data.seasalt_indices(idx_salt_offset);
    cflux[mass_mode_idx] = 0.0;
    if (emit_this_mode[ispec]) {
      for (int iorg = 0; iorg < n_organic_species; ++iorg) {
        for (int ibin = 0; ibin < salt_nsection; ++ibin) {
          if ((data.Dg(ibin) >= data.seasalt_size_range_lo(idx_salt_offset)) &&
              (data.Dg(ibin) < data.seasalt_size_range_hi(idx_salt_offset))) {
            // should use dry size, convert from number to mass flux (kg/m2/s)
            cflux_tmp = fi[ibin] * ocean_frac * data.seasalt_emis_scale_factor *
                        (4.0 / 3.0) * Constants::pi *
                        mam4::pow(data.rdry[ibin], 3) * seasalt_density;
            // Mixing state 3: internal mixture, add OM to mass and number
            // and avoid division by zero
            if (om_seasalt[ibin] > 0.0) {
              cflux[mass_mode_idx] +=
                  cflux_tmp * mass_frac_bub_section[iorg][ibin] /
                  om_seasalt[ibin] * (1.0 / (1.0 - om_seasalt[ibin]) - 1.0);
            } // if (om_seasalt)
          }   // if (Dg)
        }     // for (ibin)
      }       // for (iorg)
    }         // if (emit_this_mode)
  }           // end for (ispec)
} // end subroutine calc_marine_organic_massflux

KOKKOS_INLINE_FUNCTION
void marine_organic_emissions(
    // in
    const Real (&fi)[salt_nsection], const Real &ocean_frac,
    const SeasaltEmissionsData &data,
    const bool (&emit_this_mode)[organic_num_modes],
    // inout
    Real (&cflux)[pcnst]) {
  // input
  // fi: sea salt number fluxes in each size bin [#/m2/s]
  // ocean_frac: ocean fraction [unitless]
  // output
  // cflux: mass and number emission fluxes for aerosols [kg/m2/s or #/m2/s]

  // NOTE: these are currently passed in via SeasaltEmissionsData
  // directly to calc_org_matter_seasalt

  // for Burrows et al. (2014) organic mass fraction
  // NOTE: the mapping between these variables and those listed in the netCDF
  //       input are:
  //       'chla:CHL1','mpoly:TRUEPOLYC','mprot:TRUEPROTC','mlip:TRUELIPC',
  //       and it appears that 'chla' is unused

  // NOTE: this seems like an inefficient way to handle this
  // instead, we'll just pass the scalar variables into the function
  // as arguments, but I'll keep this here for now, just in case
  // for (int ifield = 0; ifield < n_ocean_data; ++ifield) {
  //   switch (field_name) {
  //   case mpoly:
  //     break;
  //   case mprot:
  //     break;
  //   case mlip:
  //     break;
  //   default:
  //     // FIXME: this should likely be handled in a smarter way than this.
  //     //        In the fortran code, it prints to the log,
  //     // 'Unknown field name ' fieldname' in ocean_data fields ...'
  //     break;
  //   }
  // } // end for (ifield)

  Real mass_frac_bub_section[n_organic_species_max][salt_nsection] = {{0.0}};
  Real om_seasalt[salt_nsection] = {0.0};

  // Calculate marine organic aerosol mass fraction based on
  // Burrows et al., ACP (2013)
  calc_org_matter_seasalt(
      // in
      data,
      // out
      mass_frac_bub_section, om_seasalt);

  // Calculate emission of MOM mass.

  // Determine which modes to emit MOM in depending on mixing state assumption
  // OM modes (m in this loop)
  // m = 1 => accu       (internal w/ SS)
  // m = 2 => Aitken     (internal w/ SS)
  // m = 3 => accu MOM   (external)
  // m = 4 => Aitken MOM (external)
  // Total external mixture: emit only in modes 4, 5
  calc_marine_organic_numflux(
      // in
      fi, ocean_frac, om_seasalt, emit_this_mode, data,
      // inout
      cflux);
  calc_marine_organic_massflux(
      // in
      fi, ocean_frac, om_seasalt, mass_frac_bub_section, emit_this_mode, data,
      // inout
      cflux);
} // end marine_organic_emissions()

KOKKOS_INLINE_FUNCTION
void aero_model_emissions(
    // in
    OnlineEmissionsData &online_emiss_data, SeasaltEmissionsData &seasalt_data,
    DustEmissionsData &dust_data,
    // inout
    // NOTE: fortran: cam_in%cflx
    Real (&cflux)[pcnst]) {
  // srf_temp: sea surface temperature [K]
  // sea salt number fluxes in each size bin [#/m2/s]

  // NOTE: fortran: cam_in%dstflx
  Real dust_flux_in[dust_nflux_in];
  for (int i = 0; i < dust_nflux_in; ++i) {
    dust_flux_in[i] = online_emiss_data.dust_flux_in[i];
  }
  // NOTE: fortran: cam_in%sst
  const Real surface_temp = online_emiss_data.surface_temp;
  // NOTE: fortran: state%u(:ncol,pver), state%v(:ncol,pver),
  // state%zm(:ncol,pver)
  // => u_bottom, v_bottom, z_bottom
  const Real u_bottom = online_emiss_data.u_bottom;
  const Real v_bottom = online_emiss_data.v_bottom;
  const Real z_bottom = online_emiss_data.z_bottom;
  // NOTE: fortran: cam_in%ocnfrac
  const Real ocean_frac = online_emiss_data.ocean_frac;

  Real fi[salt_nsection];
  const Real soil_erodibility = online_emiss_data.soil_erodibility;

  const Real dust_density_ = dust_density;

  dust_emis(
      // in
      dust_density_, dust_flux_in, dust_data, soil_erodibility,
      // inout
      cflux);

  init_seasalt(seasalt_data);

  calculate_seasalt_numflux_in_bins(
      // in
      surface_temp, u_bottom, v_bottom, z_bottom, seasalt_data.consta,
      seasalt_data.constb,
      // out
      fi);

  seasalt_emis(
      // in
      fi, ocean_frac, seasalt_data,
      //  inout
      cflux);

  // BAD CONSTANTS
  constexpr bool emit_this_mode[om_num_modes] = {true, true, false};

  marine_organic_emissions(
      // in
      fi, ocean_frac, seasalt_data, emit_this_mode,
      // inout
      cflux);

} // end aero_model_emissions()

KOKKOS_INLINE_FUNCTION
void aero_model_emissions(const Real &sst, const Real &ocnfrac,
                          const Real &u_bottom, const Real &v_bottom,
                          const Real &z_bottom, const const_view_1d &dstflx,
                          const Real &soil_erodibility, const Real &mpoly,
                          const Real &mprot, const Real &mlip,
                          const Real &dust_emis_scale_factor,
                          const Real &seasalt_emis_scale_factor,
                          // inout
                          view_1d &cflux_) {

  // Populate online emission data structure with inputs
  OnlineEmissionsData online_emiss_data;
  for (int idst = 0; idst < dust_nflux_in; ++idst) {
    online_emiss_data.dust_flux_in[idst] = dstflx(idst);
  }
  online_emiss_data.surface_temp = sst;
  online_emiss_data.u_bottom = u_bottom;
  online_emiss_data.v_bottom = v_bottom;
  online_emiss_data.ocean_frac = ocnfrac;
  online_emiss_data.z_bottom = z_bottom;
  online_emiss_data.soil_erodibility = soil_erodibility;

  // Populate Seasalt emissions data structure with inputs
  SeasaltEmissionsData seasalt_data;
  seasalt_data.mpoly = mpoly;
  seasalt_data.mprot = mprot;
  seasalt_data.mlip = mlip;
  seasalt_data.seasalt_emis_scale_factor = seasalt_emis_scale_factor;

  DustEmissionsData dust_data;

  // initialize dust_dmt_vwr data
  init_dust_dmt_vwr(dust_data);

  dust_data.dust_emis_scale_factor = dust_emis_scale_factor;

  // Copy 1d view to an array
  Real cflux[pcnst];
  for (int i = 0; i < pcnst; ++i) {
    cflux[i] = cflux_(i);
  }
  // Compute online emissions from dust and sea salt
  aero_model_emissions(online_emiss_data, seasalt_data, dust_data,
                       cflux); // out

  // Update cflux_ 1d view with the updated cflux
  for (int i = 0; i < pcnst; ++i) {
    cflux_(i) = cflux[i];
  }
} // end aero_model_emissions()
} // namespace mam4::aero_model_emissions
#endif
