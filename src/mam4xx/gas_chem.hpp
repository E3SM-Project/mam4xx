#ifndef MAM4XX_GAS_CHEMISTRY_HPP
#define MAM4XX_GAS_CHEMISTRY_HPP

#include "gas_chem_mechanism.hpp"
#include "mam4_math.hpp"

namespace mam4 {
namespace gas_chemistry {

enum class ImpSolOutcome {
  Converged,
  InvalidInput,
  NonfiniteResult,
  UnsafeDenominator,
  NegativeResult
};

struct ImpSolResult {
  ImpSolOutcome outcome = ImpSolOutcome::InvalidInput;

  KOKKOS_INLINE_FUNCTION
  bool success() const {
    return outcome == ImpSolOutcome::Converged;
  }
};

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
  // Provenance: user-defined legacy EAM mo_usrrxt.F90 expression; no
  // literature citation is recorded there.
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
  // adjrxt later forms lambda5 = k5*[OH]; SO2 receives 0.5*lambda5*q_DMS.
  // Provenance: user-defined legacy EAM mo_usrrxt.F90 expression; no
  // literature citation is recorded there.
  if (usr_DMS_OH_ndx > 0) {
    const Real ko =
        one + 5.5e-31 * mam4::exp(7460. / temperature) * mtot * 0.21;
    rxt[usr_DMS_OH_ndx] =
        1.7e-42 * mam4::exp(7810. / temperature) * mtot * 0.21 / ko;
  }

  // R3: SO2 + OH -> H2SO4
  // fc = 3.0e-31*(300/T)^3.3, ko = fc*M/(1 + fc*M/1.5e-12),
  // k3(T,M) = ko * 0.6^(1/(1 + log10(fc*M/1.5e-12)^2)).
  // adjrxt later forms lambda3 = k3*[OH], which transfers SO2 to H2SO4.
  // Provenance: user-defined legacy EAM mo_usrrxt.F90 expression. That source
  // explicitly marks the reference as unknown and says it is not Liao.
  if (usr_SO2_OH_ndx > 0) {
    const Real fc = 3.0e-31 * mam4::pow(300. / temperature, 3.3);
    const Real ko = fc * mtot / (one + fc * mtot / 1.5e-12);
    rxt[usr_SO2_OH_ndx] =
        ko *
        mam4::pow(0.6, one / (one + square(mam4::log10(fc * mtot / 1.5e-12))));
  }
} // usrrxt

// Solve one scalar row of (I - dt*A) q_new = q_old + dt*source. The diagonal
// of A is nonpositive for the current mechanism's nonnegative loss rates.
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

template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
imp_sol(VectorType &base_sol, const Real reaction_rates[rxntot],
        const Real het_rates[gas_pcnst], const Real extfrc[extcnt],
        const Real delt, Real prod_out[clscnt4], Real loss_out[clscnt4],
        ImpSolResult &result) {
  static_assert(gas_pcnst == 31 && clscnt4 == 30 && nzcnt == 32 &&
                    rxntot == 7 && extcnt == 9,
                "The direct gas solver requires the current MAM4xx mechanism");
  constexpr auto clsmap_4 = gas_chemistry::clsmap_4;
  constexpr auto permute_4 = gas_chemistry::permute_4;

  result = ImpSolResult{};
  for (int k = 0; k < clscnt4; ++k) {
    prod_out[k] = 0;
    loss_out[k] = 0;
  }

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
  // The ordered formulas use this exact current class layout. Fail explicitly
  // if a future generated mechanism changes the mapping without redesign.
  for (int k = 0; k < clscnt4; ++k) {
    valid_input = valid_input && clsmap_4[k] == k + 1 && permute_4[k] == k;
  }
  if (!valid_input) {
    return;
  }

  Real independent[clscnt4] = {};
  Real matrix[nzcnt] = {};
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

  Real solution[clscnt4] = {};
  for (int k = 0; k < clscnt4; ++k) {
    solution[permute_4[k]] = base_sol[clsmap_4[k]];
  }

  // Frozen reaction_rates have these meanings in the equations below:
  //   r0 = J(H2O2), supplied by the photolysis module;
  //   r1 = k1*[HO2]^2/M, the completed R1 H2O2 source;
  //   r2 = k2*[OH], r3 = k3*[OH], r4 = k4*[OH],
  //   r5 = k5*[OH], and r6 = k6*[NO3].
  // DMS -> SO2 -> H2SO4 are the only coupled rows, so solve that chain in
  // dependency order. Here h_j denotes het_rates[j] and b_k an independent
  // source. Each line solves its stated ODE with one backward-Euler step.

  // DMS losses (R4, R5, and R6):
  //   DMS + OH  -> SO2                    at lambda4 = r4
  //   DMS + OH  -> 0.5 SO2 + 0.5 HO2     at lambda5 = r5
  //   DMS + NO3 -> SO2 + HNO3             at lambda6 = r6
  // ODE: dq_DMS/dt = -(r4 + r5 + r6 + h4) q_DMS.
  ImpSolOutcome outcome = solve_backward_euler_row(
      solution[3], independent[3], matrix[5], delt, solution[3]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // SO2 production and loss:
  //   DMS -> SO2 contributions are (r4 + 0.5*r5 + r6) q_DMS;
  //   SO2 + OH -> H2SO4 removes SO2 at lambda3 = r3.
  // ODE: dq_SO2/dt = b2 + (r4 + 0.5*r5 + r6) q_DMS
  //                   - (r3 + h3) q_SO2.
  const Real so2_source = independent[2] + matrix[4] * solution[3];
  outcome = solve_backward_euler_row(solution[2], so2_source, matrix[3], delt,
                                     solution[2]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // H2SO4 production from R3: SO2 + OH -> H2SO4 at lambda3 = r3.
  // ODE: dq_H2SO4/dt = r3 q_SO2 - h2 q_H2SO4.
  const Real h2so4_source = independent[1] + matrix[2] * solution[2];
  outcome = solve_backward_euler_row(solution[1], h2so4_source, matrix[1],
                                     delt, solution[1]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // H2O2 chemistry:
  //   H2O2 + hv -> products not retained   at J(H2O2) = r0
  //   HO2 + HO2 -> H2O2                    at completed source r1
  //   H2O2 + OH -> H2O + HO2               at lambda2 = r2
  // ODE: dq_H2O2/dt = r1 - (r0 + r2 + h1) q_H2O2.
  outcome = solve_backward_euler_row(solution[0], independent[0], matrix[0],
                                     delt, solution[0]);
  if (outcome != ImpSolOutcome::Converged) {
    result.outcome = outcome;
    return;
  }

  // SOAG and the remaining aerosol-mass/modal-number entries have no chemical
  // coupling in this mechanism: dq_j/dt = b_(j-1) - h_j q_j.
  for (int k = 4; k < clscnt4; ++k) {
    outcome = solve_backward_euler_row(solution[k], independent[k],
                                       matrix[k + 2], delt, solution[k]);
    if (outcome != ImpSolOutcome::Converged) {
      result.outcome = outcome;
      return;
    }
  }

  Real trial[gas_pcnst] = {};
  for (int j = 0; j < gas_pcnst; ++j) {
    trial[j] = base_sol[j];
  }
  for (int k = 0; k < clscnt4; ++k) {
    trial[clsmap_4[k]] = solution[permute_4[k]];
  }

  Real production[clscnt4] = {};
  Real loss[clscnt4] = {};
  Real final_production[clscnt4] = {};
  imp_prod_loss(production, loss, trial, reaction_rates, het_rates);
  for (int k = 0; k < clscnt4; ++k) {
    const int m = permute_4[k];
    final_production[k] = production[m] + independent[m];
    if (!Kokkos::isfinite(final_production[k]) ||
        !Kokkos::isfinite(loss[m])) {
      result.outcome = ImpSolOutcome::NonfiniteResult;
      return;
    }
  }

  // Publish only after every row and diagnostic has been checked.
  for (int k = 0; k < clscnt4; ++k) {
    base_sol[clsmap_4[k]] = trial[clsmap_4[k]];
    prod_out[k] = final_production[k];
    loss_out[k] = loss[permute_4[k]];
  }
  result.outcome = ImpSolOutcome::Converged;
}

} // namespace gas_chemistry
} // namespace mam4
#endif
