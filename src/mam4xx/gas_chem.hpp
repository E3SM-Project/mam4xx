#ifndef MAM4XX_GAS_CHEMISTRY_HPP
#define MAM4XX_GAS_CHEMISTRY_HPP

#include "gas_chem_mechanism.hpp"
#include "mam4_math.hpp"

namespace mam4 {
namespace gas_chemistry {

enum class ImpSolOutcome {
  Converged,         // The full requested chemistry timestep was completed.
  InvalidInput,      // Invalid timestep, nonfinite input, or negative rate.
  NonfiniteResult,   // A calculation produced NaN or infinity.
  UnsafeDenominator, // A backward-Euler denominator was zero or negative.
  NegativeResult    // A solved implicit state entry was negative.
};

struct ImpSolResult {
  ImpSolOutcome outcome = ImpSolOutcome::InvalidInput;

  KOKKOS_INLINE_FUNCTION
  bool success() const {
    return outcome == ImpSolOutcome::Converged;
  }
};

// temperature is in K; mtot and invariants contain number densities in
// molecules cm^-3. The three expressions below return effective bimolecular
// coefficients in cm^3 molecule^-1 s^-1, before adjrxt applies oxidant densities.
KOKKOS_INLINE_FUNCTION
void usrrxt(Real rxt[rxntot], // inout
            const Real temperature, const Real invariants[nfs], const Real mtot,
            const int usr_HO2_HO2_ndx, const int usr_DMS_OH_ndx,
            const int usr_SO2_OH_ndx, const int inv_h2o_ndx) {

  const Real one = 1.0;

  // R1: HO2 + HO2 -> H2O2
  // k1(T,M,H2O) = (3.5e-13*exp(430/T) + 1.7e-33*M*exp(1000/T))
  //                * (1 + 1.4e-21*[H2O]*exp(2200/T)).
  // adjrxt later forms the completed H2O2 source k1*[HO2]^2/M.
  // ko and kinf are additive contributions to k1 [cm^3 molecule^-1 s^-1];
  // fc is the dimensionless enhancement due to water vapor.
  // Rate expression matches Sander et al. (2006), JPL Publication 06-2,
  // Table 1-1 and note B13 (pp. 1-9, 1-44--1-45); see references in
  // gas_chem_mechanism.hpp. The evaluation also includes O2 as a product;
  // only H2O2 production is retained in this reduced mechanism.
  // B13 attributes the water enhancement to Lii, Sauer, and Gordon (1981),
  // J. Phys. Chem. 85, 2833-2834, and Kircher and Sander (1984),
  // J. Phys. Chem. 88, 2082-2091. The code was ported from mo_usrrxt.F90.
  if (usr_HO2_HO2_ndx > 0) {
    const Real ko = 3.5e-13 * mam4::exp(430.0 / temperature);
    const Real kinf = 1.7e-33 * mtot * mam4::exp(1000. / temperature);
    const Real fc = one + 1.4e-21 * invariants[inv_h2o_ndx] *
                              mam4::exp(2200. / temperature);
    rxt[usr_HO2_HO2_ndx] = (ko + kinf) * fc;
  }

  // R5: DMS + OH -> 0.5 SO2 + 0.5 HO2
  // k5(T,M) = 1.7e-42*exp(7810/T)*M*0.21
  //             / (1 + 5.5e-31*exp(7460/T)*M*0.21).
  // The literal 0.21 is the legacy mechanism's fixed O2 mixing fraction.
  // M*0.21 is the prescribed O2 number density; ko is dimensionless.
  // adjrxt later forms lambda5 = k5*[OH]; SO2 receives 0.5*lambda5*q_DMS.
  // Ported from legacy EAM mo_usrrxt.F90. JPL Publication 19-5, Table 1I,
  // note I20, evaluates the OH-addition channel but recommends a different
  // rate expression. A literature source for this legacy fit and its lumped
  // 0.5 SO2 yield has not been verified; I20 is not a citation for either.
  // JPL 06-2, note I19, also used [O2] in the denominator, but with different
  // coefficients. That shared form does not establish this fit's provenance.
  if (usr_DMS_OH_ndx > 0) {
    const Real ko =
        one + 5.5e-31 * mam4::exp(7460. / temperature) * mtot * 0.21;
    rxt[usr_DMS_OH_ndx] =
        1.7e-42 * mam4::exp(7810. / temperature) * mtot * 0.21 / ko;
  }

  // R3: SO2 + OH -> H2SO4
  // fc = 3.0e-31*(300/T)^3.3, ko = fc*M/(1 + fc*M/1.5e-12),
  // k3(T,M) = ko * 0.6^(1/(1 + log10(fc*M/1.5e-12)^2)).
  // fc is the low-pressure coefficient [cm^6 molecule^-2 s^-1]; fc*M,
  // ko, and the high-pressure limit 1.5e-12 have units cm^3 molecule^-1 s^-1.
  // The ratio inside log10 and the broadening multiplier are dimensionless.
  // adjrxt later forms lambda3 = k3*[OH], which transfers SO2 to H2SO4.
  // Rate coefficients and the 0.6 falloff expression match DeMore et al.
  // (1997), JPL Publication 97-4, Table 2, note I4 and the table's falloff
  // formula (p. 126). That evaluation describes OH + SO2 + M -> HOSO2 + M;
  // H2SO4 here is the lumped downstream product, not the elementary product.
  // I4 cites Wine et al. (1984), J. Phys. Chem. 88, 2095-2104,
  // doi:10.1021/j150654a031, among the underlying experimental datasets.
  // The implemented coefficients are an evaluated synthesis, not a direct
  // transcription of that single laboratory study.
  if (usr_SO2_OH_ndx > 0) {
    const Real fc = 3.0e-31 * mam4::pow(300. / temperature, 3.3);
    const Real ko = fc * mtot / (one + fc * mtot / 1.5e-12);
    rxt[usr_SO2_OH_ndx] =
        ko *
        mam4::pow(0.6, one / (one + square(mam4::log10(fc * mtot / 1.5e-12))));
  }
} // usrrxt

// Advance one equation, dq/dt = source + diagonal*q, with backward Euler:
// q_new = (q_old + dt*source) / (1 - dt*diagonal).
// old and updated use the same work-entry unit; source uses that unit s^-1,
// diagonal is in s^-1, and dt is in s. The denominator is dimensionless.
// Here diagonal is minus the total first-order loss coefficient. A coupled
// source must use the already-solved end-of-step value of its upstream species.
// imp_sol checks the timestep and inputs before calling this helper; it keeps
// updated in private storage until every species and diagnostic passes checks.
KOKKOS_INLINE_FUNCTION
ImpSolOutcome solve_backward_euler_row(const Real old, const Real source,
                                       const Real diagonal, const Real dt,
                                       Real &updated) {
  if (!Kokkos::isfinite(source)) {
    return ImpSolOutcome::NonfiniteResult;
  }
  const Real denominator = 1 - dt * diagonal;
  if (!Kokkos::isfinite(denominator)) {
    return ImpSolOutcome::NonfiniteResult;
  }
  if (denominator <= 0) {
    return ImpSolOutcome::UnsafeDenominator;
  }
  const Real numerator = old + dt * source;
  if (!Kokkos::isfinite(numerator)) {
    return ImpSolOutcome::NonfiniteResult;
  }
  updated = numerator / denominator;
  if (!Kokkos::isfinite(updated)) {
    return ImpSolOutcome::NonfiniteResult;
  }
  if (updated < 0) {
    return ImpSolOutcome::NegativeResult;
  }
  return ImpSolOutcome::Converged;
}

// Update work-array entries 1 through 30 for one full chemistry timestep.
// Entry 0 is O3 and is left unchanged. All rates and forcing are fixed during
// this call. On failure, base_sol is unchanged and prod_out/loss_out are zero.
// On success, prod_out/loss_out are rates at the new state, not time integrals.
// Units and indexing:
// - base_sol[0..5]: gas molar mixing ratios (mol tracer/mol dry air in EAMxx),
//   not number concentrations in molecules cm^-3.
// - base_sol[6..30]: aerosol-mass and modal-number entries in the caller's
//   legacy converted work-array units, not additional gas mole fractions.
//   This solver preserves those conversions; it does not redefine their units.
// - reaction_rates[0] and [2..6], and het_rates[0..30]: s^-1 coefficients;
//   reaction_rates[1]: H2O2 mixing-ratio source per second.
// - extfrc[0..8]: already normalized sources in the destination entry unit s^-1.
// - delt: seconds; prod_out[k] and loss_out[k]: base_sol[k+1] unit s^-1.
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
imp_sol(VectorType &base_sol, const Real reaction_rates[rxntot],
        const Real het_rates[gas_pcnst], const Real extfrc[extcnt],
        const Real delt, Real prod_out[clscnt4], Real loss_out[clscnt4],
        ImpSolResult &result) {
  static_assert(gas_pcnst == 31 && clscnt4 == 30 && nzcnt == 32 &&
                    rxntot == 7 && extcnt == 9,
                "Cannot compile imp_sol: expected 31 state entries (O3 plus "
                "30 updated entries), 32 matrix coefficients, 7 reaction "
                "rates, and 9 forcing values. Update the backward-Euler solver "
                "if the chemistry mechanism changes.");
  // The formulas below use work-array indices directly: O3 is entry 0, and
  // implicit unknown k is entry k+1, with no reordering. Check the generated
  // species maps at compile time so a changed mapping cannot silently make
  // these formulas update the wrong species.
  static_assert([]() constexpr {
    for (int k = 0; k < clscnt4; ++k) {
      if (clsmap_4[k] != k + 1 || permute_4[k] != k) {
        return false;
      }
    }
    return true;
  }(), "Cannot compile imp_sol: the species maps must select state entries "
       "1-30 in their array order, excluding O3 at entry 0. Update the "
       "solver indexing to match the changed species mapping.");

  result = ImpSolResult{};
  for (int k = 0; k < clscnt4; ++k) {
    prod_out[k] = 0;
    loss_out[k] = 0;
  }

  // State and external forcing may be signed; the solved endpoints must be
  // nonnegative. Reaction coefficients and losses for entries 1-30 must be
  // nonnegative. O3's heterogeneous rate is not used by this solver.
  bool valid_input = Kokkos::isfinite(delt) && delt > 0;
  for (int j = 0; j < gas_pcnst; ++j) {
    valid_input = valid_input && Kokkos::isfinite(base_sol[j]) &&
                  Kokkos::isfinite(het_rates[j]) &&
                  (j == o3_idx || het_rates[j] >= 0);
  }
  for (int i = 0; i < rxntot; ++i) {
    valid_input = valid_input && Kokkos::isfinite(reaction_rates[i]) &&
                  reaction_rates[i] >= 0;
  }
  for (int i = 0; i < extcnt; ++i) {
    valid_input = valid_input && Kokkos::isfinite(extfrc[i]);
  }
  if (!valid_input) {
    return;
  }

  Real independent[clscnt4] = {};
  Real matrix[nzcnt] = {};
  // indprd constructs sources that do not depend on the updated state;
  // linmat constructs loss coefficients and DMS->SO2->H2SO4 couplings.
  indprd(4, independent, reaction_rates, extfrc);
  linmat(matrix, reaction_rates, het_rates);
  for (int k = 0; k < clscnt4; ++k) {
    if (!Kokkos::isfinite(independent[k])) {
      result.outcome = ImpSolOutcome::NonfiniteResult;
      return;
    }
  }
  for (int i = 0; i < nzcnt; ++i) {
    if (!Kokkos::isfinite(matrix[i])) {
      result.outcome = ImpSolOutcome::NonfiniteResult;
      return;
    }
  }

  // Entries 1-5 are H2O2, H2SO4, SO2, DMS, and SOAG; entries 6-30 are aerosol
  // mass and modal particle numbers. Keep updated values private so a failed
  // calculation leaves base_sol unchanged. Copy checked values back only
  // after the whole step succeeds.
  Real trial[gas_pcnst] = {};
  for (int j = 0; j < gas_pcnst; ++j) {
    trial[j] = base_sol[j];
  }

  // reaction_rates is fixed for the entire timestep. In the equations below,
  // q_X is the gas mixing ratio, M is air number density [molecules cm^-3],
  // and [OH], [HO2], and [NO3] are prescribed densities [molecules cm^-3]:
  //   r0 = J(H2O2) [s^-1], supplied by the photolysis module;
  //   r1 = k1*[HO2]^2/M, the completed R1 H2O2 mixing-ratio source [s^-1];
  //   r2 = k2*[OH], r3 = k3*[OH], r4 = k4*[OH],
  //   r5 = k5*[OH], and r6 = k6*[NO3], all first-order coefficients [s^-1].
  // Solve DMS first, then use its new value to solve SO2, then use the new
  // SO2 value to solve H2SO4. These are the only inter-species dependencies.
  // h_j = het_rates[j] is the first-order removal coefficient [s^-1] for
  // work entry j; b_k = independent[k] is the source for implicit unknown k
  // (work entry k+1), in that entry's unit s^-1. Every equation uses one
  // backward-Euler step over the full timestep.

  // DMS losses (R4, R5, and R6):
  //   DMS + OH  -> SO2                    at lambda4 = r4
  //   DMS + OH  -> 0.5 SO2 + 0.5 HO2     at lambda5 = r5
  //   DMS + NO3 -> SO2 + HNO3             at lambda6 = r6
  // ODE: dq_DMS/dt = -(r4 + r5 + r6 + h4) q_DMS.
  ImpSolOutcome outcome = solve_backward_euler_row(
      trial[4], independent[3], matrix[5], delt, trial[4]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // SO2 production and loss:
  //   DMS -> SO2 contributions are (r4 + 0.5*r5 + r6) q_DMS;
  //   SO2 + OH -> H2SO4 removes SO2 at lambda3 = r3.
  // ODE: dq_SO2/dt = b2 + (r4 + 0.5*r5 + r6) q_DMS
  //                   - (r3 + h3) q_SO2.
  const Real so2_source = independent[2] + matrix[4] * trial[4];
  outcome = solve_backward_euler_row(trial[3], so2_source, matrix[3], delt,
                                     trial[3]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // H2SO4 production from R3: SO2 + OH -> H2SO4 at lambda3 = r3.
  // ODE: dq_H2SO4/dt = r3 q_SO2 - h2 q_H2SO4.
  const Real h2so4_source = independent[1] + matrix[2] * trial[3];
  outcome = solve_backward_euler_row(trial[2], h2so4_source, matrix[1],
                                     delt, trial[2]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // H2O2 chemistry:
  //   H2O2 + hv -> products not retained   at J(H2O2) = r0
  //   HO2 + HO2 -> H2O2                    at completed source r1
  //   H2O2 + OH -> H2O + HO2               at lambda2 = r2
  // ODE: dq_H2O2/dt = r1 - (r0 + r2 + h1) q_H2O2.
  outcome = solve_backward_euler_row(trial[1], independent[0], matrix[0],
                                     delt, trial[1]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // SOAG, aerosol-mass entries, and modal particle-number entries each have
  // only external forcing and heterogeneous removal in this chemistry step:
  // dq_j/dt = b_(j-1) - h_j q_j. None depends on another updated species.
  for (int k = 4; k < clscnt4; ++k) {
    outcome = solve_backward_euler_row(trial[k + 1], independent[k],
                                       matrix[k + 2], delt, trial[k + 1]);
    if (outcome != ImpSolOutcome::Converged) {
      result.outcome = outcome;
      return;
    }
  }

  Real production[clscnt4] = {};
  Real loss[clscnt4] = {};
  // Evaluate instantaneous production and loss at the completed new state.
  // Include the state-independent sources before checking the output rates.
  imp_prod_loss(production, loss, trial, reaction_rates, het_rates);
  for (int k = 0; k < clscnt4; ++k) {
    production[k] = production[k] + independent[k];
    if (!Kokkos::isfinite(production[k]) ||
        !Kokkos::isfinite(loss[k])) {
      result.outcome = ImpSolOutcome::NonfiniteResult;
      return;
    }
  }

  // All 30 updated entries and their output rates are valid. Commit them
  // together; leave O3 untouched. Earlier returns leave outputs zero and
  // preserve every entry of base_sol.
  for (int k = 0; k < clscnt4; ++k) {
    base_sol[k + 1] = trial[k + 1];
    prod_out[k] = production[k];
    loss_out[k] = loss[k];
  }
  result.outcome = ImpSolOutcome::Converged;
}

} // namespace gas_chemistry
} // namespace mam4
#endif
