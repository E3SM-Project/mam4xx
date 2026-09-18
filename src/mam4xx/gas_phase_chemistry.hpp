#ifndef MAM4XX_MICROPHYSICS_GAS_CHEMISTRY_HPP
#define MAM4XX_MICROPHYSICS_GAS_CHEMISTRY_HPP

#include "gas_chem.hpp"
#include "mo_photo.hpp"

namespace mam4 {

namespace microphysics {
// ================================================================
//  Gas Phase Chemistry
// ================================================================
// number of gases+aerosols species
using mam4::gas_chemistry::gas_pcnst;
// number of species with external forcing
using mam4::gas_chemistry::extcnt;
// number of invariants
using mam4::gas_chemistry::nfs;
// number of chemical reactions
using mam4::gas_chemistry::rxntot;
// number of photolysis reactions
using mam4::mo_photo::phtcnt;
// index of total atm density in invariant array
// FIXME: Hardwired indices, should be obtained from a config file
constexpr int ndx_h2so4 = 2;
constexpr int synoz_ndx = -1;

// Following (indices (ndxes?) are taken from mam4 validation data and
// translated from
// 1-based indices to 0-based indices)
constexpr int usr_HO2_HO2_ndx = 1, usr_DMS_OH_ndx = 5, usr_SO2_OH_ndx = 3,
              inv_h2o_ndx = 3;

// index of total atm density in invariant array
using mam4::gas_chemistry::indexm;

// performs gas phase chemistry calculations on a single level of a single
// atmospheric column
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void gas_phase_chemistry(
    // in
    const Real temp, const Real dt,
    const Real photo_rates[mam4::mo_photo::phtcnt], const Real extfrc[extcnt],
    const Real invariants[nfs], const Real het_rates[gas_pcnst],
    // out
    VectorType &qq, gas_chemistry::ImpSolResult &result) {

  //=====================================================================
  // ... set rates for "tabular" and user specified reactions
  //=====================================================================
  Real reaction_rates[rxntot];
  mam4::gas_chemistry::setrxt(reaction_rates, // out
                              temp);          // in

  // set reaction rates based on chemical invariants
  mam4::gas_chemistry::usrrxt(reaction_rates,                       // out
                              temp, invariants, invariants[indexm], // in
                              usr_HO2_HO2_ndx, usr_DMS_OH_ndx,      // in
                              usr_SO2_OH_ndx,                       // in
                              inv_h2o_ndx);                         // in

  mam4::gas_chemistry::adjrxt(reaction_rates,                  // out
                              invariants, invariants[indexm]); // in

  //===================================
  // Photolysis rates at time = t(n+1)
  //===================================

  // compute the rate of change from forcing
  Real extfrc_rates[extcnt]; // [1/cm^3/s]
  for (int mm = 0; mm < extcnt; ++mm) {
    if (mm != synoz_ndx) {
      extfrc_rates[mm] = extfrc[mm] / invariants[indexm];
    }
  }

  // save h2so4 before gas phase chem (for later new particle nucleation)
  Real del_h2so4_gasprod = qq[ndx_h2so4];

  //===========================
  // Class solution algorithms
  //===========================

  // copy photolysis rates into reaction_rates (assumes photolysis rates come
  // first)
  for (int i = 0; i < phtcnt; ++i) {
    reaction_rates[i] = photo_rates[i];
  }

  // ... solve for "Implicit" species
  using mam4::gas_chemistry::itermax;
  bool factor[itermax];
  for (int i = 0; i < itermax; ++i) {
    factor[i] = true;
  }

  // initialize error tolerances
  using mam4::gas_chemistry::clscnt4;
  Real epsilon[clscnt4];
  mam4::gas_chemistry::imp_slv_inti(epsilon);

  // solve chemical system implicitly
  Real prod_out[clscnt4], loss_out[clscnt4];
  mam4::gas_chemistry::imp_sol(qq,                                      // out
                               reaction_rates, het_rates, extfrc_rates, // in
                               dt, factor,                              // in
                               epsilon, prod_out, loss_out, result);    // out

  // save h2so4 change by gas phase chem (for later new particle nucleation)
  if (ndx_h2so4 > 0) {
    del_h2so4_gasprod = qq[ndx_h2so4] - del_h2so4_gasprod;
  }
}

} // namespace microphysics
} // namespace mam4
#endif
