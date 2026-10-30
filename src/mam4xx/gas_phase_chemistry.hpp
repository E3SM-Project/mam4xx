#ifndef MAM4XX_MICROPHYSICS_GAS_CHEMISTRY_HPP
#define MAM4XX_MICROPHYSICS_GAS_CHEMISTRY_HPP

#include "gas_chem.hpp"
#include "mo_photo.hpp"

namespace mam4 {

namespace microphysics {
// ================================================================
//  Gas Phase Chemistry
// ================================================================
// number of gas, aerosol-mass, and modal-aerosol-number species
using mam4::gas_chemistry::gas_pcnst;
// number of species with external forcing
using mam4::gas_chemistry::extcnt;
// number of invariants
using mam4::gas_chemistry::nfs;
// number of chemical reactions
using mam4::gas_chemistry::rxntot;
// number of photolysis reactions
using mam4::mo_photo::phtcnt;
constexpr int synoz_ndx = -1;

// Indices used by usrrxt to write reaction coefficients into reaction_rates
// and read water-vapor number density from invariants.
constexpr int usr_HO2_HO2_ndx = 1, usr_DMS_OH_ndx = 5, usr_SO2_OH_ndx = 3,
              inv_h2o_ndx = 3;

// index of total air number density [molecules cm^-3] in the invariant array
using mam4::gas_chemistry::indexm;

// performs gas phase chemistry calculations on a single level of a single
// atmospheric column
// temp: K; dt: s; photo_rates: photolysis frequencies [s^-1].
// invariants: prescribed number densities [molecules cm^-3], ordered as
// M, N2, O2, H2O, OH, NO3, HO2, cnst_O3; none is updated by this call.
// extfrc: unnormalized external forcing (gas sources in molecules cm^-3 s^-1).
// het_rates: first-order removal coefficients [s^-1], one per work entry.
// qq: in/out chemistry work array; gas entries 0-5 are molar mixing ratios
// (mol tracer/mol dry air in EAMxx), not molecules cm^-3. Aerosol-mass and
// modal-number entries 6-30 retain the caller's legacy work-array conversions;
// their precise physical units should not be inferred from the gas entries.
// Review needed: reconcile the legacy modal-number unit/conversion conventions.
// The direct solver publishes state only after the full step succeeds. The
// legacy solver publishes each accepted internal substep but never a failed
// trial. Legacy failures return their status in every build. Direct failures
// abort with a reason-specific message when NDEBUG is absent; with NDEBUG,
// they return failure and leave the complete input state unchanged. Callers
// must inspect the returned status and record unsuccessful intervals.
// The returned counts and intervals describe actual accepted work: a failed
// direct call has one failed attempt, no cuts, no accepted steps, and zero
// accepted time. Successful direct calls accept one step covering dt seconds.
template <typename VectorType>
KOKKOS_INLINE_FUNCTION gas_chemistry::ImpSolResult gas_phase_chemistry(
    // in
    const Real temp, const Real dt,
    const Real photo_rates[mam4::mo_photo::phtcnt], const Real extfrc[extcnt],
    const Real invariants[nfs], const Real het_rates[gas_pcnst],
    // inout state
    VectorType &qq) {

  //=====================================================================
  // Calculate temperature-dependent reaction coefficients. Prescribed
  // oxidant densities below turn them into rates held fixed for this step.
  //=====================================================================
  Real reaction_rates[rxntot];
  mam4::gas_chemistry::setrxt(reaction_rates, // out
                              temp);          // in

  // Supply the three custom rate expressions from the legacy EAM mechanism.
  mam4::gas_chemistry::usrrxt(reaction_rates,                       // out
                              temp, invariants, invariants[indexm], // in
                              usr_HO2_HO2_ndx, usr_DMS_OH_ndx,      // in
                              usr_SO2_OH_ndx,                       // in
                              inv_h2o_ndx);                         // in

  // Multiply coefficients by prescribed OH or NO3 densities, and form the
  // HO2+HO2 source. These oxidants are not updated by the gas solver.
  mam4::gas_chemistry::adjrxt(reaction_rates,                  // out
                              invariants, invariants[indexm]); // in

  //===================================
  // Use photolysis supplied for this chemistry step; it is not recomputed here.
  //===================================

  // Normalize external forcing by air number density. For gases, this converts
  // molecules cm^-3 s^-1 / molecules cm^-3 to mixing-ratio s^-1. Aerosol and
  // modal-number sources must use the corresponding legacy work-entry basis.
  Real extfrc_rates[extcnt]; // normalized work-entry tendencies per second
  for (int mm = 0; mm < extcnt; ++mm) {
    extfrc_rates[mm] = extfrc[mm] / invariants[indexm];
  }

  //===========================
  // Integrate the 30 implicit work entries; O3 remains unchanged.
  //===========================

  // J1: H2O2 + hv -> products not retained by this reduced mechanism.
  // photo_rates[0] is the upstream photolysis frequency J(H2O2) [s^-1], so
  // the H2O2 loss used by imp_sol is J(H2O2) * q_H2O2. Actinic-flux and
  // cross-section calculations remain in the photolysis module.
  // mo_photo::jlong sums cross section * quantum yield * actinic photon flux
  // over wavelength bins above 200 nm; "long" does not mean thermal infrared.
  // Actinic-flux basis: Madronich (1987), JGR 92, 9740-9752,
  // doi:10.1029/JD092iD08p09740. H2O2 absorption measurements include Lin,
  // Rohatgi, and DeMore (1978), GRL 5, 113-115, doi:10.1029/GL005i002p00113.
  // These references do not identify the exact XSQY/RSF input-table generation.
  for (int i = 0; i < phtcnt; ++i) {
    reaction_rates[i] = photo_rates[i];
  }

  using mam4::gas_chemistry::clscnt4;
  Real prod_out[clscnt4], loss_out[clscnt4];

#if defined(MAM4XX_GAS_CHEMISTRY_SOLVER_LEGACY)
  // Apply the corrected PR #586 legacy solver. Each accepted internal substep
  // is committed, while a failed trial is discarded. Preserve legacy behavior
  // by returning failures without aborting, including any accepted substeps.
  bool factor[gas_chemistry::legacy::itermax];
  for (int i = 0; i < gas_chemistry::legacy::itermax; ++i) {
    factor[i] = true;
  }
  Real epsilon[clscnt4];
  gas_chemistry::legacy::imp_slv_inti(epsilon);
  return gas_chemistry::legacy::imp_sol(qq, reaction_rates, het_rates,
                                        extfrc_rates, dt, factor, epsilon,
                                        prod_out, loss_out);
#else
  // Solve one full backward-Euler timestep with fixed rates and forcing. The
  // direct solver leaves qq unchanged on failure.
  const auto outcome = mam4::gas_chemistry::imp_sol(
      qq, reaction_rates, het_rates, extfrc_rates, dt, prod_out, loss_out);
  gas_chemistry::ImpSolResult result;
  result.outcome = outcome;
  result.requested_interval = dt;
  if (outcome == gas_chemistry::ImpSolOutcome::Converged) {
    result.accepted_steps = 1;
    result.accepted_interval = dt;
    return result;
  }
  result.failed_attempts = 1;
#ifndef NDEBUG
  switch (outcome) {
  case gas_chemistry::ImpSolOutcome::InvalidInput:
    Kokkos::abort("gas_phase_chemistry: invalid input; chemistry state was "
                  "left unchanged");
  case gas_chemistry::ImpSolOutcome::NonfiniteResult:
    Kokkos::abort("gas_phase_chemistry: a gas-chemistry calculation produced "
                  "NaN or infinity; chemistry state was left unchanged");
  case gas_chemistry::ImpSolOutcome::UnsafeDenominator:
    Kokkos::abort("gas_phase_chemistry: the backward-Euler denominator was "
                  "zero or negative; chemistry state was left unchanged");
  case gas_chemistry::ImpSolOutcome::NegativeResult:
    Kokkos::abort("gas_phase_chemistry: the backward-Euler update produced a "
                  "negative state entry; chemistry state was left unchanged");
  default:
    Kokkos::abort("gas_phase_chemistry: unknown direct solver outcome; "
                  "chemistry state was left unchanged");
  }
#endif
  return result;
#endif
}

} // namespace microphysics
} // namespace mam4
#endif
