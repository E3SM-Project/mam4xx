#ifndef MAM4XX_MODAL_AERO_OPT_HPP
#define MAM4XX_MODAL_AERO_OPT_HPP

#include <Kokkos_Complex.hpp>

#include "aero_config.hpp"
#include "mam4_math.hpp"
#include "modal_aero_calcsize.hpp"
#include "ndrop.hpp"
#include "water_uptake.hpp"

namespace mam4 {
namespace modal_aero_opt {

using View1D = DeviceType::view_1d<Real>;
using View2D = DeviceType::view_2d<Real>;
using View3D = DeviceType::view_3d<Real>;
using ComplexView2D = DeviceType::view_2d<Kokkos::complex<Real>>;
using ComplexView1D = DeviceType::view_1d<Kokkos::complex<Real>>;
using View5D = Kokkos::View<Real *****>;
using View0D = Kokkos::View<Real>;

using CalcsizeData = modal_aero_calcsize::CalcsizeData;
constexpr int pver = mam4::nlev;
constexpr int ntot_amode = mam4::AeroConfig::num_modes();
// FIXME:  is top_lev equal to 1 in aerosol optics ?
constexpr int top_lev = 0;
//
//  min, max aerosol surface mode radius treated [m]
constexpr Real rmmin = 0.01e-6; // BAD CONSTANT
constexpr Real rmmax = 25.e-6;  // BAD CONSTANT

// From radconstants
constexpr int nswbands = 14;
constexpr int nlwbands = 16;

// Dimension sizes in coefficient arrays used to parameterize aerosol radiative
// properties in terms of refractive index and wet radius
constexpr int ncoef = 5;
constexpr int prefr = 7;
constexpr int prefi = 10;

// BAD CONSTANT
constexpr Real small_value_40 = 1.e-40;
constexpr Real neg_small_value_16 = -1.e-16;

// Density of liquid water (STP)
constexpr Real rhoh2o = Constants::density_h2o;
//  reciprocal of gravit
constexpr Real rga = 1.0 / Constants::gravity;
// FIXME: this values may cause differences because of number of digits in the
// Boltzmann's constant and Avogadro's number
constexpr Real rair = Constants::r_gas_dry_air;

//  These are indices to the band for diagnostic output
// Fortran to C++ indexing
constexpr int idx_sw_diag = 9; // index to sw visible band (10 in Fortran)
constexpr int idx_nir_diag =
    7; // index to sw near infrared (778-1240 nm) band(8 in Fortran
constexpr int idx_uv_diag =
    10; // index to sw uv (345-441 nm) band (11 in Fortran)

// FIXME; is this values set somewhere else?
constexpr int max_nspec = 7;

// dimensions for reading refractive index from modal radioactive properties
// files
constexpr int coef_number = 5;   // number of coefficients
constexpr int refindex_real = 7; // real refractive index
constexpr int refindex_im = 10;  // imaginary refractive index

struct AerosolOpticsDeviceData {
  // devices views
  // FIXME: add description of these tables.
  View1D refitabsw[ntot_amode][nswbands];
  View1D refrtabsw[ntot_amode][nswbands];
  View3D abspsw[ntot_amode][nswbands];
  View3D asmpsw[ntot_amode][nswbands];

  View1D refrtablw[ntot_amode][nlwbands];
  View1D refitablw[ntot_amode][nlwbands];
  View3D absplw[ntot_amode][nlwbands];
  View3D extpsw[ntot_amode][nswbands];

  ComplexView1D crefwlw;
  ComplexView1D crefwsw;
  ComplexView2D specrefindex_sw[ntot_amode];
  ComplexView2D specrefindex_lw[ntot_amode];
};

inline void set_aerosol_optics_data_for_modal_aero_sw_views(
    AerosolOpticsDeviceData &aersol_optics_data) {

  for (int d1 = 0; d1 < ntot_amode; ++d1) {
    for (int d5 = 0; d5 < nswbands; ++d5) {
      aersol_optics_data.abspsw[d1][d5] =
          View3D("abspsw", coef_number, refindex_real, refindex_im);
      aersol_optics_data.extpsw[d1][d5] =
          View3D("extpsw", coef_number, refindex_real, refindex_im);
      aersol_optics_data.asmpsw[d1][d5] =
          View3D("asmpsw", coef_number, refindex_real, refindex_im);

      aersol_optics_data.refrtabsw[d1][d5] = View1D("refrtabsw", refindex_real);

      aersol_optics_data.refitabsw[d1][d5] = View1D("refitabsw", refindex_im);
    } // d5
  }   // d1

} // configure_aerosol_optics_data

inline void set_aerosol_optics_data_for_modal_aero_lw_views(
    AerosolOpticsDeviceData &aersol_optics_data) {

  for (int d1 = 0; d1 < ntot_amode; ++d1)
    for (int d5 = 0; d5 < nlwbands; ++d5) {
      aersol_optics_data.absplw[d1][d5] =
          View3D("absplw3", coef_number, refindex_real, refindex_im);
      aersol_optics_data.refrtablw[d1][d5] = View1D("refrtablw", refindex_real);
      aersol_optics_data.refitablw[d1][d5] = View1D("refitablw", refindex_im);
    } // d5

} // set_aerosol_optics_data_for_modal_aero_lw_views

inline void
set_complex_views_modal_aero(AerosolOpticsDeviceData &aersol_optics_data) {
  for (int i = 0; i < ntot_amode; ++i) {
    aersol_optics_data.specrefindex_sw[i] =
        ComplexView2D("specrefindex_sw", max_nspec, nswbands);
  }

  for (int i = 0; i < ntot_amode; ++i) {
    aersol_optics_data.specrefindex_lw[i] =
        ComplexView2D("specrefindex_lw", max_nspec, nlwbands);
  }

  aersol_optics_data.crefwlw = ComplexView1D("crefwlw", nlwbands);
  aersol_optics_data.crefwsw = ComplexView1D("crefwsw", nswbands);
} // set_complex_views_modal_aero

// reshape specrefindex and copy to device
inline void set_device_specrefindex(
    const ComplexView2D specrefindex[ntot_amode], const std::string &wave_type,
    const ComplexView2D::host_mirror_type &specrefndx_host) {

  std::string view_name;
  int nbands = 0;

  if (wave_type == "short_wave") {
    view_name = "specrefindex_sw";
    nbands = nswbands;
  } else if (wave_type == "long_wave") {
    view_name = "specrefindex_lw";
    nbands = nlwbands;
  } else {
    printf(
        "wave type does not exit. Only two options: short_wave or long_wave");
    exit(1);
  } // end wave_type

  ComplexView2D::host_mirror_type specrefindex_host(view_name, max_nspec,
                                                    nbands);

  for (int mm = 0; mm < ntot_amode; ++mm) {
    const int nspec = AeroConfig::nspec_amode(mm);
    for (int ibands = 0; ibands < nbands; ++ibands) {
      // // Fortran to C++ indexing
      for (int ll = 0; ll < nspec; ++ll) {
        specrefindex_host(ll, ibands) =
            specrefndx_host(ibands, AeroConfig::lspectype_amode(ll, mm));
      } // ll
    }   // ibands
    Kokkos::deep_copy(specrefindex[mm], specrefindex_host);

  } // mm

}; // set_device_specrefindex

KOKKOS_INLINE_FUNCTION
void modal_size_parameters(const Real sigma_logr_aer,
                           const Real dgnumwet, // in
                           Real &radsurf, Real &logradsurf, Real cheb[ncoef],
                           const bool ismethod2) {

  /* ncol
   sigma_logr_aer   geometric standard deviation of number distribution
   dgnumwet    aerosol wet number mode diameter [m]
   radsurf     aerosol surface mode radius [m]
   logradsurf  mam4::log(aerosol surface mode radius)
   cheb      chebychev polynomial parameters

  FORTRAN refactoring: ismethod is tempararily used to ensure BFB test
  bool ismethod2
  */
  constexpr Real half = 0.5;
  constexpr Real one = 1.0;
  constexpr Real two = 2.0;
  const Real xrmin = mam4::log(rmmin);
  const Real xrmax = mam4::log(rmmax);

  const Real alnsg_amode = mam4::log(sigma_logr_aer);
  const Real explnsigma = mam4::exp(two * alnsg_amode * alnsg_amode);
  // do kk = top_lev, pver
  // do icol = 1, ncol
  //  convert from number mode diameter to surface area
  radsurf = half * dgnumwet * explnsigma;
  //  --------------- FORTRAN refactoring -------------------
  //  here two calculations are used to ensure passing BFB test
  //  can be simplified (there is only round-off difference
  if (ismethod2) {
    logradsurf = mam4::log(half * dgnumwet) + two * alnsg_amode * alnsg_amode;
  } else {
    logradsurf = mam4::log(radsurf);

  } // ismethod2
  //  --------------- FORTRAN refactoring -------------------

  //  normalize size parameter
  Real xrad = mam4::utils::min_max_bound(xrmin, xrmax, logradsurf);
  xrad = (two * xrad - xrmax - xrmin) / (xrmax - xrmin);
  // chebyshev polynomials
  cheb[0] = one;
  cheb[1] = xrad;
  for (int nc = 2; nc < ncoef; ++nc) {
    cheb[nc] = two * xrad * cheb[nc - 1] - cheb[nc - 2];
  } // nc

} // modal_size_parameters

KOKKOS_INLINE_FUNCTION
void calc_parameterized(const Real coef[ncoef], const Real cheb_k[ncoef],
                        Real &para) {
  /* calculate parameterized absorption, extinction or asymmetry factor
  further calculations are needed. see modal_aero_sw and modal_aero_lw
  coef(ncoef)
  cheb_k(ncoef)
  para */
  constexpr Real half = 0.5;
  para = half * coef[0];
  for (int nc = 1; nc < ncoef; ++nc) {
    // printf("cheb_k[nc] %e coef[nc] %e \n", cheb_k[nc],  coef[nc]);
    para += cheb_k[nc] * coef[nc];
  }

} // calc_parameterized

KOKKOS_INLINE_FUNCTION
void update_aod_spec(const Real scath2o,
                     const Real absh2o, // in
                     const Real sumhygro, const Real sumscat,
                     const Real sumabs, // in
                     const Real hygro_s, const Real palb,
                     const Real dopaer, // in
                     Real &scat_s, Real &abs_s, Real &aod_s) {

  /* scat_s, abs_s, aod_s   scatering, absorption and aod for a species
  update aerosol optical depth from scattering and absorption*/
  constexpr Real one = 1.0;
  scat_s = (scat_s + scath2o * hygro_s / sumhygro) / sumscat;
  abs_s = (abs_s + absh2o * hygro_s / sumhygro) / sumabs;
  const Real aodc = (abs_s * (one - palb) + palb * scat_s) * dopaer;
  aod_s += aodc;

} // update_aod_spec

KOKKOS_INLINE_FUNCTION
void calc_volc_ext(const int trop_level, const ConstColumnView &state_zm,
                   const ColumnView &ext_cmip6_sw_m, const ColumnView &extinct,
                   Real &tropopause_m) {
  /* calculate contributions from volcanic aerosol extinction
  trop_level(pcols)tropopause level for each column
  state_zm(:,:)  state%zm [m]
  ext_cmip6_sw(pcols,pver)  aerosol shortwave extinction [1/m]
  extinct(pcols,pver)  aerosol extinction [1/m]
  tropopause_m(pcols)  tropopause height [m]
  kk_tropp = trop_level(icol) */
  constexpr Real half = 0.5;
  // diagnose tropopause height
  tropopause_m = state_zm(trop_level); // in meters
  // update tropopause layer first
  // Note: Multiplication by km_inv_to_m_inv
  extinct(trop_level) =
      half * (extinct(trop_level) + ext_cmip6_sw_m(trop_level));
  // extinction is assigned read in values only for visible band above
  // tropopause
  for (int kk = 0; kk < trop_level; ++kk) {
    extinct(kk) = ext_cmip6_sw_m(kk);
  }
} // calc_volc_ext

KOKKOS_INLINE_FUNCTION
void calc_diag_spec(const Real specmmr_k, const Real mass_k, const Real vol,
                    const Real specrefr, const Real specrefi,
                    const Real hygro_aer, Real &burden_s, Real &scat_s,
                    Real &abs_s, Real &hygro_s) {
  /* calculate some diagnostics for a species
  specmmr_k(:)   mmr at level kk [kg/kg]
  mass_k(:)         mass at layer kk [kg/m2]
  vol(:)  volume concentration of aerosol species [m3/kg]
  specrefr, specrefi   real and image part of specrefindex
  hygro_aer        aerosol hygroscopicity [unitless]
  burden_s(pcols)  aerosol burden of species [kg/m2]
  scat_s(pcols)    scattering of species [unitless]
  abs_s(pcols)     absorption of species [unit?]
  hygro_s(pcols)   hygroscopicity of species [unitless]*/
  constexpr Real zero = 0;

  burden_s = zero;
  burden_s += specmmr_k * mass_k;
  scat_s = vol * specrefr;
  abs_s = -vol * specrefi;
  hygro_s = vol * hygro_aer;

} // calc_diag_spec

KOKKOS_INLINE_FUNCTION
void calc_refin_complex(const int lwsw, const int ilwsw, const Real qaerwat_kk,
                        const Real *specvol, const ComplexView2D &specrefindex,
                        const int nspec, const ComplexView1D &crefwlw,
                        const ComplexView1D &crefwsw, Real &dryvol,
                        Real &wetvol, Real &watervol,
                        Kokkos::complex<Real> &crefin, Real &refr, Real &refi) {
  /*-------------------------------------------------------------------
calculate complex refractive index
also output wetvol and watervol
-------------------------------------------------------------------*/

  /* lwsw    indicator if this is lw or sw lw =0 and sw =1
  qaerwat_kk(:)    aerosol water at level kk [g/g]
  specvol(:,:)     volume concentration of aerosol specie [m3/kg]
  complex(r8), intent(in) :: specrefindex(:,:)      species refractive index
  dryvol(pcols)     volume concentration of aerosol mode [m3/kg]
  wetvol(pcols)     volume concentration of wet mode [m3/kg]
  watervol(pcols)   volume concentration of water in each mode [m3/kg]
  refr(pcols)       real part of refractive index
  refi(pcols)       imaginary part of refractive index
  complex(r8),intent(out) :: crefin(pcols)  complex refractive index

  refractive index for water read in read_water_refindex
  crefwsw(nswbands)  complex refractive index for water visible
  crefwlw(nlwbands)  complex refractive index for water infrared

  FIXME
   if ((lwsw /= 'lw') .and. (lwsw /= 'sw')) then
       call endrun('calc_refin_complex is called with '// lwsw// ', it should
       be called with either lw or sw')
   endif
  */

  // crefin(:ncol) = (0._r8, 0._r8)
  constexpr Real zero = 0;
  dryvol = zero;

  crefin = {};

  for (int i = 0; i < nspec; ++i) {
    dryvol += specvol[i];
    crefin += specvol[i] * specrefindex(i, ilwsw);
  }

  watervol = qaerwat_kk / rhoh2o;
  wetvol = watervol + dryvol;
  if (watervol < zero) // lwsw=='lw'
  {
    // BAD CONSTANT
    // FIXME
    // if (abs(watervol > 1.e-1*wetvol))
    // {
    // 	write(iulog,*) 'watervol,wetvol,dryvol=',watervol(icol), &
    //                  wetvol(icol),dryvol(icol)
    // }
    watervol = zero;
    wetvol = dryvol;
  } // end if watervol < zero && lwsw=='lw'

  //  some different treatments for lw and sw
  if (lwsw == 0) // lwsw=='lw'
  {
    crefin += watervol * crefwlw(ilwsw);
    // BAD CONSTANT
    if (wetvol > small_value_40) {
      crefin /= wetvol;
    } // end if wetvol(icol) > small_value_40

  } else if (lwsw == 1) //  lwsw=='sw
  {

    crefin += watervol * crefwsw(ilwsw);
    // BAD CONTANT
    crefin /= mam4::max(wetvol, small_value_40);

  } // lwsw=='lw'
  // FIXME
  refr = crefin.real();
  refi = abs(crefin.imag());

} // calc_refin_complex

KOKKOS_INLINE_FUNCTION
void compute_factors(const int prefri, const Real ref_ind,
                     const Real *ref_table, int &ix, Real &tt) {
  // Compute factors for the real or imaginary parts

  // prefri, ncol
  // ref_table(:) refractive index table [unitless]
  // ref_ind(:)   refractive index       [unitless]

  // FORTRAN refactor note: "do_print" is kept to maintain code consistenty with
  // the original code. THis can be removed from the C++ ported code
  //  logical, intent(in), optional :: do_print  to print log msg or not

  // intent-inouts/outs
  // ix(:)
  // tt(:)
  // BAD CONSTANT
  constexpr Real threshold = 1.e-20;
  constexpr Real zero = 0;
  ix = 1;
  tt = zero;
  if (prefri > 1) {

    int ii = 0;
    for (ii = 0; ii < prefri; ++ii) {
      if (ref_ind < ref_table[ii]) {
        break;
      }
    } // ii
    // FIXME: check Fortran to C++ indexing conversion
    ix = mam4::max(ii - 1, 0);
    const int ip1 = mam4::min(ix + 1, prefri - 1);
    const Real dx = ref_table[ip1] - ref_table[ix];
    if (abs(dx) > threshold) {
      tt = (ref_ind - ref_table[ix]) / dx;
      // FIXME: I did not port the following:
      // if (present(do_print) .and. do_print) then
      //       if(tt(ic) < 0._r8 .or. tt(ic) > 1._r8) then
      //          write(iulog,*)
      //          'tt,ref_ind,ix,ref_table,dx=',tt(ic),ref_ind(ic),ix(ic),ref_table(ix(ic)),dx
      //       endif
      //    endif

    } // (abs(dx) > threshold)

  } // if(prefri > 1)

} // compute_factors

KOKKOS_INLINE_FUNCTION
void binterp(const View3D &table, const Real ref_real, const Real ref_img,
             const Real ref_real_tab[prefr], const Real ref_img_tab[prefi],
             int &itab, int &jtab, Real &ttab, Real &utab, Real coef[ncoef],
             const int itab_1) {
  /*------------------------------------------------------------------------------
   Bilinear interpolation along the refractive index dimensions
   of the table to estimate Chebyshev coefficients at an
   intermediate refractive index.

   In short wave, the first call computes itab, jtab, ttab, utab and coef.
   The subsequent calls use itab, jtab, ttab and utab as inputs and compute coef

   In long wave, we have just one call to compute itab,jtab,ttab, utab and coef
  ------------------------------------------------------------------------------*/

  // intent-ins
  // ncol
  // table(ncoef,prefr,prefi)
  // ref_real(pcols), ref_img(pcols) real and imganinary parts of refractive
  // indices [unitless] ref_real_tab(prefr), ref_img_tab(prefi) real and
  // imganinary table refractive indices [unitless]

  // intent-inouts/outs
  // itab(pcols), jtab(pcols)
  // ttab(pcols), utab(pcols)
  // coef(pcols,ncoef) coefficient interpolated bilinearly
  constexpr Real one = 1.0;
  if (itab_1 <= 0.0) {
    // compute factors for the real part
    compute_factors(prefr, ref_real, ref_real_tab, itab, ttab);

    // compute factors for the imaginary part
    compute_factors(prefi, ref_img, ref_img_tab, jtab, utab);
  } // itab_1 < -1

  const Real tu = ttab * utab;
  const Real tuc = ttab - tu;
  const Real tcuc = one - tuc - utab;
  const Real tcu = utab - tu;
  const int jp1 = mam4::min(jtab + 1, prefi - 1);
  const int ip1 = mam4::min(itab + 1, prefr - 1);
  for (int icoef = 0; icoef < ncoef; ++icoef) {
    coef[icoef] = tcuc * table(icoef, itab, jtab) +
                  tuc * table(icoef, ip1, jtab) + tu * table(icoef, ip1, jp1) +
                  tcu * table(icoef, itab, jp1);
  } // icoef

} // binterp
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void compute_calcsize_and_water_uptake_dr(
    const Real &pmid, const Real &temperature, Real &cldn,
    const VectorType &state_q_kk, // in
    const VectorType &qqcw_k,     // in
    const Real &dt, const CalcsizeData &calcsizedata,
    // outputs
    Real dgnumwet_m_kk[ntot_amode], Real qaerwat_m_kk[ntot_amode]) {

  Real dgncur_c_kk[ntot_amode] = {};
  Real dgnumdry_m_kk[ntot_amode] = {};
  Real ptend[pcnst] = {};
  Real dqqcwdt[pcnst] = {};
  //  Calculate aerosol size distribution parameters and aerosol water uptake
  // For prognostic aerosols
  modal_aero_calcsize::modal_aero_calcsize_sub(state_q_kk, // in
                                               qqcw_k,     // in/out
                                               dt, calcsizedata, dgnumdry_m_kk,
                                               dgncur_c_kk, ptend, dqqcwdt);

  mam4::water_uptake::modal_aero_water_uptake_dr(state_q_kk, temperature, pmid,
                                                 cldn, dgnumdry_m_kk,
                                                 dgnumwet_m_kk, qaerwat_m_kk);
} // compute_calcsize_water_uptake_dr

template <typename VectorType>
KOKKOS_INLINE_FUNCTION void modal_aero_sw_wo_diagnostics_k(
    const Real &pdeldry, const Real &pmid, const Real &temperature, Real &cldn,
    const VectorType &state_q_kk, // in
    const VectorType &qqcw_k,     // in
    const Real &dt, const AerosolOpticsDeviceData &aersol_optics_data,
    const CalcsizeData &calcsizedata,
    // outputs
    const View2D &tauxar, const View2D &wa, const View2D &ga,
    const View2D &fa) {

  const Real xrmax = mam4::log(rmmax);
  //  calculates aerosol sw radiative properties
  // dt               timestep [s]
  // lchnk             chunk id
  // ncol              number of active columns in the chunk
  //  state_q(:,:,:)    water and tracers (state%q) in state [kg/kg]
  //  state_zm(:,:)     mid-point height (state%zm) [m]
  //  temperature(:,:)  temperature [K]
  //  pmid(:,:)         mid-point pressure [Pa]
  //  pdel(:,:)         pressure interval [Pa]
  //  pdeldry(:,:)      dry mass pressure interval [Pa]

  //  cldn(:,:)          layer cloud fraction [fraction]

  //  nnite           number of night columns
  //  idxnite(nnite)  local column indices of night columns
  //  trop_level(pcols)tropopause level for each column
  //  ext_cmip6_sw(pcols,pver)  aerosol shortwave extinction [1/m]
  //  is_cmip6_volc

  //  qqcw(:)                      Cloud borne aerosols mixing ratios [kg/kg or
  //  1/kg] tauxar(ntot_amode,nswbands)  layer extinction optical depth [1]
  //  wa(ntot_amode,nswbands)      layer single-scatter albedo [1]
  //  ga(ntot_amode,nswbands)      asymmetry factor [1]
  //  fa(ntot_amode,nswbands)      forward scattered fraction [1]

  //  Local variables
  // real(r8),    pointer :: specmmr(:,:)         species mass mixing ratio
  // [kg/kg] spectype             species type hygro_aer
  // hygroscopicity [1]

  // sigma_logr_aer          geometric standard deviation of number
  // distribution radsurf(pcols,pver)     aerosol surface mode radius
  // logradsurf(pcols,pver)  mam4::log(aerosol surface mode radius)
  // cheb(ncoef,pcols,pver)  chebychev polynomial parameters

  // specvol(:,:)         volume concentration of aerosol specie [m3/kg]
  // specdens(:)          species density for all species [kg/m3]
  // specrefindex(:,:)      species refractive index

  // refr(pcols)      real part of refractive index
  // refi(pcols)      imaginary part of refractive index
  // crefin(pcols)    complex refractive index

  // dryvol(pcols)    volume concentration of aerosol mode [m3/kg]
  // watervol(pcols)  volume concentration of water in each mode [m3/kg]
  // wetvol(pcols)    volume concentration of wet mode [m3/kg]

  // pext(pcols)      parameterized specific extinction [m2/kg]

  // pabs(pcols)      parameterized specific absorption [m2/kg]
  // pasm(pcols)      parameterized asymmetry factor [unitless?]
  // palb(pcols)      parameterized single scattering albedo [unitless]

  constexpr Real zero = 0.0;
  constexpr Real one = 1.0;

  const Real mass = pdeldry * rga;

  Real cheb_kk[ncoef] = {};
  Real specvol[max_nspec] = {};
  // Real mean_std_dev_nmodes[ntot_amode] = {};
  Real dgnumwet_m_kk[ntot_amode] = {};
  Real qaerwat_m_kk[ntot_amode] = {};
  compute_calcsize_and_water_uptake_dr(
      pmid, temperature, cldn, state_q_kk, qqcw_k, dt, // in
      calcsizedata, dgnumwet_m_kk, qaerwat_m_kk);

  for (int mm = 0; mm < ntot_amode; ++mm) {
    //  get mode info
    const int nspec = AeroConfig::nspec_amode(mm);
    // const Real sigma_logr_aer = sigmag_amode[mm];
    // CHECK if mean_std_dev_nmodes is equivalent to sigmag_amode
    const Real sigma_logr_aer = calcsizedata.mean_std_dev_nmodes[mm];

    Real logradsurf = 0;
    Real radsurf = 0;
    modal_size_parameters(sigma_logr_aer, dgnumwet_m_kk[mm], // in
                          radsurf, logradsurf, cheb_kk, false);

    for (int isw = 0; isw < nswbands; ++isw) {

      for (int ll = 0; ll < nspec; ++ll) {

        // get aerosol properties and save for each species
        // Fortran to C++ indexing
        auto specmmr = state_q_kk[AeroConfig::lmassptr_amode(ll, mm)];
        // FIXME: move specdens to init
        //  Fortran to C++ indexing
        const Real specdens =
            AeroConfig::specdens_amode(AeroConfig::lspectype_amode(ll, mm));

        // allocate(specvol(pcols,nspec),stat=istat)
        specvol[ll] = specmmr / specdens;
      } // ll

      // lw =0 and sw =1
      Real dryvol, wetvol, watervol = {};
      // NOTE: The following logic exists only in MAM4xx,
      // it doesn't exists in Fortan MAM4. This logic is added
      // to deal with very small (essentially zero)
      // negative values for dryvol
      dryvol = zero;
      for (int ispec = 0; ispec < nspec; ++ispec) {
        dryvol += specvol[ispec];
      }
      if (dryvol < zero && dryvol > neg_small_value_16) {
        // if dryvol is negative and a very small number,
        // set all optical properties to zero
        // and continue to next iteration
        tauxar(mm, isw) = zero;
        wa(mm, isw) = zero;
        ga(mm, isw) = zero;
        fa(mm, isw) = zero;
        continue; // continue to next isw
      }

      Kokkos::complex<Real> crefin = {};
      Real refr, refi = {};

      calc_refin_complex(1, isw, qaerwat_m_kk[mm], specvol,
                         aersol_optics_data.specrefindex_sw[mm], nspec,
                         aersol_optics_data.crefwlw, aersol_optics_data.crefwsw,
                         dryvol, wetvol, watervol, crefin, refr, refi);

      // interpolate coefficients linear in refractive index
      // first call calcs itab,jtab,ttab,utab

      const auto sub_extpsw = aersol_optics_data.extpsw[mm][isw];
      const auto ref_real_tab = aersol_optics_data.refrtabsw[mm][isw];
      const auto ref_img_tab = aersol_optics_data.refitabsw[mm][isw];

      int itab = zero;   // index for Bilinear interpolation
      int itab_1 = zero; //  index for Bilinear interpolation column 1
      // FIXME: part of binterp only apply for first column
      int jtab = zero;
      Real ttab, utab = {}; // coef for Bilinear interpolation
      Real cext[ncoef], cabs[ncoef],
          casm[ncoef] = {}; //   coefficient for extinction, absoption, and
                            //  asymmetry [unitless]

      binterp(sub_extpsw, refr, refi, ref_real_tab.data(), ref_img_tab.data(),
              itab, jtab, ttab, utab, cext, itab_1);

      const auto sub_abspsw = aersol_optics_data.abspsw[mm][isw];

      binterp(sub_abspsw, refr, refi, ref_real_tab.data(), ref_img_tab.data(),
              itab, jtab, ttab, utab, cabs, itab_1);

      const auto sub_asmpsw = aersol_optics_data.asmpsw[mm][isw];

      binterp(sub_asmpsw, refr, refi, ref_real_tab.data(), ref_img_tab.data(),
              itab, jtab, ttab, utab, casm, itab_1);

      // parameterized optical properties
      Real pext = zero; //    parameterized specific extinction [m2/kg]
      calc_parameterized(cext, cheb_kk, pext);
      Real pabs = zero; // parameterized specific absorption [m2/kg]
      calc_parameterized(cabs, cheb_kk, pabs);

      Real pasm = zero; // parameterized asymmetry factor [unitless?]
      calc_parameterized(casm, cheb_kk, pasm);

      //  do icol=1,ncol

      if (logradsurf <= xrmax) {
        pext = mam4::exp(pext);
      } else {
        // BAD CONSTANT
        pext = 1.5 / (radsurf * rhoh2o); //  geometric optics
      }                                  // if logradsurf(kk) <= xrmax

      // convert from m2/kg water to m2/kg aerosol
      // FIXME: specpext is used by check_error_warning, which is not ported
      // yet. const Real specpext = pext;// specific extinction [m2/kg]
      pext *= wetvol * rhoh2o;
      pabs *= wetvol * rhoh2o;
      pabs = mam4::utils::min_max_bound(zero, pext, pabs);
      Real palb =
          one -
          pabs /
              mam4::max(pext, small_value_40); // parameterized single
                                               // scattering albedo [unitless]
      const Real dopaer = pext * mass; // aerosol optical depth in layer [1]

      // end cols
      tauxar(mm, isw) = dopaer;
      wa(mm, isw) = dopaer * palb;
      ga(mm, isw) = dopaer * palb * pasm;
      fa(mm, isw) = dopaer * palb * pasm * pasm;

    } // isw
  }   // mm
} // modal_aero_sw_wo_diagnostics_k ends

inline int get_work_len_aerosol_optics() {
  // tauxar, wa, ga, fa
  // Note:
  return 4 * pver * ntot_amode * nswbands;
}

KOKKOS_INLINE_FUNCTION
void modal_aero_sw(const ThreadTeam &team, const Real dt,
                   mam4::Prognostics &progs, const Atmosphere &atm,
                   const ConstColumnView &pdel, const ConstColumnView &pdeldry,
                   // const ColumnView qqcw_fld[aero_model::pcnst],
                   const View2D &tauxar, const View2D &wa, const View2D &ga,
                   const View2D &fa,
                   const AerosolOpticsDeviceData &aersol_optics_data,
                   const CalcsizeData &calcsizedata,
                   // aerosol optical depth
                   Real &aodvis, const View1D &work)

{
  const ConstColumnView temperature = atm.temperature;
  const ConstColumnView pmid = atm.pressure;
  const ConstColumnView state_zm = atm.height;
  const ConstColumnView cldn = atm.cloud_fraction;

  auto work_ptr = (Real *)work.data();
  const auto tauxar_work = View3D(work_ptr, ntot_amode, nswbands, pver);
  work_ptr += pver * ntot_amode * nswbands;
  const auto wa_work = View3D(work_ptr, ntot_amode, nswbands, pver);
  work_ptr += pver * ntot_amode * nswbands;
  const auto ga_work = View3D(work_ptr, ntot_amode, nswbands, pver);
  work_ptr += pver * ntot_amode * nswbands;
  const auto fa_work = View3D(work_ptr, ntot_amode, nswbands, pver);
  work_ptr += pver * ntot_amode * nswbands;

  constexpr Real zero = 0;

  // Need to have these defined outside kernels
  // otherwise undefined errors on cuda dev
  auto _nswbands = nswbands;
  auto _pver = pver;
  auto _top_lev = top_lev;

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, _nswbands), [&](int i) {
    // BAD CONSTANT
    tauxar(i, 0) = zero; // BAD CONSTANT
    wa(i, 0) = 0.925;    // BAD CONSTANT
    ga(i, 0) = 0.850;    // BAD CONSTANT
    fa(i, 0) = 0.7225;   // BAD CONSTANT
  });

  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 1, _pver), [&](int kk) {
    for (int i = 0; i < nswbands; ++i) {
      tauxar(i, kk) = zero;
      wa(i, kk) = zero;
      ga(i, kk) = zero;
      fa(i, kk) = zero;
    }
  });

  team.team_barrier();

  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, _top_lev, _pver), [&](int kk) {
        Real state_q[pcnst] = {};
        Real qqcw[pcnst] = {};

        utils::extract_stateq_from_prognostics(progs, atm, state_q, kk);
        utils::extract_qqcw_from_prognostics(progs, qqcw, kk);

        Real cldn_kk = cldn(kk);
        const auto tauxar_kkp =
            Kokkos::subview(tauxar_work, Kokkos::ALL(), Kokkos::ALL(), kk);
        const auto wa_kkp =
            Kokkos::subview(wa_work, Kokkos::ALL(), Kokkos::ALL(), kk);
        const auto ga_kkp =
            Kokkos::subview(ga_work, Kokkos::ALL(), Kokkos::ALL(), kk);
        const auto fa_kkp =
            Kokkos::subview(fa_work, Kokkos::ALL(), Kokkos::ALL(), kk);
        modal_aero_sw_wo_diagnostics_k(pdeldry(kk), pmid(kk), temperature(kk),
                                       cldn_kk,
                                       state_q, // in
                                       qqcw,    // in
                                       dt, aersol_optics_data, calcsizedata,
                                       // outputs
                                       tauxar_kkp, wa_kkp, ga_kkp, fa_kkp);
        utils::inject_qqcw_to_prognostics(qqcw, progs, kk);
        utils::inject_stateq_to_prognostics(state_q, progs, kk);
      });

  team.team_barrier();

  for (int isw = 0; isw < nswbands; ++isw) {

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, pver), [&](const int kk) {
          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, ntot_amode),
              [&](int imode, Real &suma) {
                suma += tauxar_work(imode, isw, kk);
              },
              tauxar(isw, kk + 1));

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, ntot_amode),
              [&](int imode, Real &suma) { suma += wa_work(imode, isw, kk); },
              wa(isw, kk + 1));

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, ntot_amode),
              [&](int imode, Real &suma) { suma += ga_work(imode, isw, kk); },
              ga(isw, kk + 1));

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, ntot_amode),
              [&](int imode, Real &suma) { suma += fa_work(imode, isw, kk); },
              fa(isw, kk + 1));
        }); // kk

  } // isw

  // compute aerosol_optical depth
  aodvis = zero;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, _top_lev, _pver),
      [&](int kk, Real &suma) {
        for (int imode = 0; imode < ntot_amode; ++imode) {
          //  aerosol species loop
          // savaervis ! true if visible wavelength (0.55 micron)
          // aodvis(icol )    = aodvis(icol) + dopaer(icol)
          // dopaer = tauxar_work
          suma += tauxar_work(imode, idx_sw_diag, kk);
        } // imode
      },
      aodvis);

} //
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void modal_aero_lw_k(
    const Real &pdeldry, const Real &pmid, const Real &temperature, Real &cldn,
    const VectorType &state_q_kk, // in
    const VectorType &qqcw_k,     // in
    const Real &dt, const AerosolOpticsDeviceData &aersol_optics_data,
    const CalcsizeData &calcsizedata,
    // outputs
    Real *tauxar) {

  // FORTRAN refactoring: For prognostic aerosols only, other options are
  // removed
  constexpr Real zero = 0;

  const auto crefwlw = aersol_optics_data.crefwlw;
  const auto crefwsw = aersol_optics_data.crefwsw;

  Real cheb_kk[ncoef] = {};
  Real specvol[max_nspec] = {};

  // layer dry mass [kg/m2]
  const Real mass = pdeldry * rga;

  // calcsize and water_uptake_dr outputs that are required by aerosol_optics
  Real dgnumwet_m_kk[ntot_amode] = {};
  Real qaerwat_m_kk[ntot_amode] = {};
  compute_calcsize_and_water_uptake_dr(
      pmid, temperature, cldn, state_q_kk, qqcw_k, dt, // in
      calcsizedata, dgnumwet_m_kk, qaerwat_m_kk);

  for (int mm = 0; mm < ntot_amode; ++mm) {

    // get mode info
    const int nspec = AeroConfig::nspec_amode(mm);
    // const Real sigma_logr_aer = sigmag_amode[mm];
    // CHECK if mean_std_dev_nmodes is equivalent to sigmag_amode
    const Real sigma_logr_aer = calcsizedata.mean_std_dev_nmodes[mm];

    // calc size parameter for all columns
    // FORTRAN refactoring: ismethod2 is tempararily used to ensure BFB test.
    // can be removed when porting to C++
    Real logradsurf = 0;
    Real radsurf = 0;
    modal_size_parameters(sigma_logr_aer, dgnumwet_m_kk[mm], // in
                          radsurf, logradsurf, cheb_kk, true);

    for (int ilw = 0; ilw < nlwbands; ++ilw) {

      for (int ll = 0; ll < nspec; ++ll) {
        // Fortran to C++ indexing
        auto specmmr = state_q_kk[AeroConfig::lmassptr_amode(ll, mm)];
        // FIXME: move specdens to int
        //  Fortran to C++ indexing
        const Real specdens =
            AeroConfig::specdens_amode(AeroConfig::lspectype_amode(ll, mm));

        specvol[ll] = specmmr / specdens;
      } // ll

      // lw =0 and sw =1
      // calculate complex refractive index
      Real dryvol = zero;
      Real wetvol = zero;
      Real watervol = zero;
      Kokkos::complex<Real> crefin = {};
      Real refr, refi = {};
      calc_refin_complex(0, ilw, qaerwat_m_kk[mm], specvol,
                         aersol_optics_data.specrefindex_lw[mm], nspec, crefwlw,
                         crefwsw, dryvol, wetvol, watervol, crefin, refr, refi);

      const auto ref_real_tab = aersol_optics_data.refrtablw[mm][ilw];
      const auto ref_img_tab = aersol_optics_data.refitablw[mm][ilw];

      // interpolate coefficients linear in refractive index
      // first call calcs itab,jtab,ttab,utab
      int itab = zero;
      int itab_1 = zero;
      int jtab = zero;
      Real ttab, utab = {};
      Real cabs[ncoef] = {};
      binterp(aersol_optics_data.absplw[mm][ilw], refr, refi,
              ref_real_tab.data(), ref_img_tab.data(), itab, jtab, ttab, utab,
              cabs, itab_1);

      // parameterized optical properties
      Real pabs = zero; //    parameterized specific extinction [m2/kg]
      calc_parameterized(cabs, cheb_kk, pabs);

      pabs *= wetvol * rhoh2o;
      pabs = mam4::max(zero, pabs);
      Real dopaer = pabs * mass;

      // update absorption optical depth
      tauxar[ilw] += dopaer;

      // FIXME: specpext is used by check_error_warning, which is not ported
      // yet.
      // FORTRAN refactor: check and writeout error/warning message
      // FORTRAN refactor: This if condition is never met in testing run ...
      // call check_error_warning('lw', icol, kk,mm, ilw, nspec, list_idx,&
      // in
      //                 dopaer, pabs(icol), dryvol, wetvol, watervol,
      //                 crefin,cabs,&  in specdens, specrefindex, specvol, &
      //                  in nerr_dopaer)  inout

    } // ilw

  } // mm}

} // kk

KOKKOS_INLINE_FUNCTION
void modal_aero_lw(const ThreadTeam &team, const Real dt,
                   mam4::Prognostics &progs, const Atmosphere &atm,
                   const ConstColumnView &pdel, const ConstColumnView &pdeldry,
                   // parameters
                   const AerosolOpticsDeviceData &aersol_optics_data,
                   const CalcsizeData &calcsizedata,
                   // output
                   const View2D &tauxar) {

  const ConstColumnView temperature = atm.temperature;
  const ConstColumnView pmid = atm.pressure;
  const ConstColumnView cldn = atm.cloud_fraction;
  // calculates aerosol lw radiative properties
  // dt        time step [s]
  // state_q(:,:,:)    water and tracers (state%q) in state [kg/kg]
  // temperature(:,:)  temperature [K]
  // pmid(:,:)         mid-point pressure [Pa]
  // pdel(:,:)         pressure interval [Pa]
  // pdeldry(:,:)      dry mass pressure interval [Pa]
  // cldn(:,:)         layer cloud fraction [fraction]
  // qqcw(:)                Cloud borne aerosols mixing ratios [kg/kg or 1/kg]
  // tauxar(pcols,pver,nlwbands)  layer absorption optical depth

  // Need to have these defined outside kernels
  // otherwise undefined errors on cuda dev
  auto _pver = pver;
  auto _top_lev = top_lev;

  constexpr Real zero = 0.0;
  // dry mass in each cell
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, _pver), [&](int kk) {
    // initialize output variables
    for (int i = 0; i < nlwbands; ++i) {
      tauxar(i, kk) = zero;
    }
  });
  team.team_barrier();

  // inputs

  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, _top_lev, _pver), [&](int kk) {
        Real cldn_kk = cldn(kk);

        Real state_q[pcnst] = {};
        Real qqcw[pcnst] = {};
        utils::extract_stateq_from_prognostics(progs, atm, state_q, kk);
        utils::extract_qqcw_from_prognostics(progs, qqcw, kk);

        Real tauxar_kkp[nlwbands] = {};
        modal_aero_lw_k(pdeldry(kk), pmid(kk), temperature(kk), cldn_kk,
                        state_q, // in
                        qqcw,    // in
                        dt, aersol_optics_data, calcsizedata,
                        // outputs
                        tauxar_kkp);

        for (int ilw = 0; ilw < nlwbands; ++ilw) {
          tauxar(ilw, kk) = tauxar_kkp[ilw];
        }
        utils::inject_qqcw_to_prognostics(qqcw, progs, kk);
        utils::inject_stateq_to_prognostics(state_q, progs, kk);
      });
} // modal_aero_lw

} // namespace modal_aero_opt

} // end namespace mam4

#endif
