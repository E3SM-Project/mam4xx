#ifndef MAM4XX_GAS_CHEMISTRY_HPP
#define MAM4XX_GAS_CHEMISTRY_HPP

#include "gas_chem_mechanism.hpp"
#include "mam4_math.hpp"

namespace mam4 {
namespace gas_chemistry {

enum class ImpSolOutcome {
  // Values 0-5 retain the existing EAMxx legacy diagnostic encoding.
  Converged = 0,
  ConvergedAfterRetry = 1,
  InvalidInput = 2,
  NonfiniteIterate = 3,
  CutLimitExhausted = 4,
  MaximumStepsExhausted = 5,
  // Direct-solver failures extend that encoding without reusing legacy codes.
  NonfiniteResult = 6,
  UnsafeDenominator = 7,
  NegativeResult = 8
};

// Accounting for one requested chemistry interval. Intervals are in seconds;
// counts are nonnegative integers. A failed direct solve accepts zero steps
// and zero time. Legacy failures may retain successful internal substeps.
struct ImpSolResult {
  ImpSolOutcome outcome = ImpSolOutcome::InvalidInput;
  int failed_attempts = 0;
  int cut_count = 0;
  int accepted_steps = 0;
  Real requested_interval = 0;
  Real accepted_interval = 0;

  KOKKOS_INLINE_FUNCTION
  bool success() const {
    const bool converged = outcome == ImpSolOutcome::Converged ||
                           outcome == ImpSolOutcome::ConvergedAfterRetry;
    return converged && Kokkos::isfinite(requested_interval) &&
           requested_interval > 0 && Kokkos::isfinite(accepted_interval) &&
           accepted_interval >= 0 &&
           mam4::abs(requested_interval - accepted_interval) <= 1.0e-4;
  }
};

// temperature is in K; mtot and invariants contain number densities in
// molecules cm^-3. The three expressions below return effective bimolecular
// coefficients in cm^3 molecule^-1 s^-1, before adjrxt applies oxidant
// densities.
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

namespace legacy {

// BAD CONSTANTs
constexpr int itermax = 11;
const Real rel_err = 1.0e-3;
// NOTE: high_rel_err is unused currently
// const Real high_rel_err = 1.0e-4;
const int max_time_steps = 1000;

using ImpSolOutcome = gas_chemistry::ImpSolOutcome;
using ImpSolResult = gas_chemistry::ImpSolResult;

// NOTE: at this point we are taking RHS units of (maybe) [1/s] to [s]
// and the units are internally consistent
KOKKOS_INLINE_FUNCTION
void lu_fac(Real lu[nzcnt]) {
  const Real one = 1;
  lu[0] = one / lu[0];
  lu[1] = one / lu[1];
  lu[3] = one / lu[3];
  lu[5] = one / lu[5];
  lu[6] = one / lu[6];
  lu[7] = one / lu[7];
  lu[8] = one / lu[8];
  lu[9] = one / lu[9];
  lu[10] = one / lu[10];
  lu[11] = one / lu[11];
  lu[12] = one / lu[12];
  lu[13] = one / lu[13];
  lu[14] = one / lu[14];
  lu[15] = one / lu[15];
  lu[16] = one / lu[16];
  lu[17] = one / lu[17];
  lu[18] = one / lu[18];
  lu[19] = one / lu[19];
  lu[20] = one / lu[20];
  lu[21] = one / lu[21];
  lu[22] = one / lu[22];
  lu[23] = one / lu[23];
  lu[24] = one / lu[24];
  lu[25] = one / lu[25];
  lu[26] = one / lu[26];
  lu[27] = one / lu[27];
  lu[28] = one / lu[28];
  lu[29] = one / lu[29];
  lu[30] = one / lu[30];
  lu[31] = one / lu[31];
} // lu_fac

// TODO: again, the units for b[1:2] look like they could be inconsistent
// lu = sys_jac [s]--mostly, maybe?; when passed in within gas_chem.hpp
// b = forcing [1/s]--maybe; when passed in from gas_chem.hpp
KOKKOS_INLINE_FUNCTION
void lu_slv(Real lu[nzcnt], Real b[clscnt4]) {
  b[29] *= lu[31];
  b[28] *= lu[30];
  b[27] *= lu[29];
  b[26] *= lu[28];
  b[25] *= lu[27];
  b[24] *= lu[26];
  b[23] *= lu[25];
  b[22] *= lu[24];
  b[21] *= lu[23];
  b[20] *= lu[22];
  b[19] *= lu[21];
  b[18] *= lu[20];
  b[17] *= lu[19];
  b[16] *= lu[18];
  b[15] *= lu[17];
  b[14] *= lu[16];
  b[13] *= lu[15];
  b[12] *= lu[14];
  b[11] *= lu[13];
  b[10] *= lu[12];
  b[9] *= lu[11];
  b[8] *= lu[10];
  b[7] *= lu[9];
  b[6] *= lu[8];
  b[5] *= lu[7];
  b[4] *= lu[6];
  b[3] *= lu[5];
  b[2] -= lu[4] * b[3];
  b[2] *= lu[3];
  // the above 2 lines are equivalent to b[2] = (b[2] - (lu[4] * b[3])) * lu[3]
  b[1] -= lu[2] * b[2];
  b[1] *= lu[1];
  // the above 2 lines are equivalent to b[1] = (b[1] - (lu[2] * b[2])) * lu[1]
  b[0] *= lu[0];
} // lu_slv

// TODO: the below calculations appear to have inconsistent units for
// mat[0, 2, 3, 5]
// NOTE: it's *possible* we could be ok, if the odd-looking sums in linmat()
// turn out to be ok
KOKKOS_INLINE_FUNCTION
void nlnmat(Real mat[nzcnt], const Real lmat[nzcnt], const Real dti) {
  mat[0] = lmat[0] - dti;
  mat[1] = lmat[1] - dti;
  mat[2] = lmat[2];
  mat[3] = lmat[3] - dti;
  mat[4] = lmat[4];
  mat[5] = lmat[5] - dti;
  mat[6] = lmat[6] - dti;
  mat[7] = lmat[7] - dti;
  mat[8] = lmat[8] - dti;
  mat[9] = lmat[9] - dti;
  mat[10] = lmat[10] - dti;
  mat[11] = lmat[11] - dti;
  mat[12] = lmat[12] - dti;
  mat[13] = lmat[13] - dti;
  mat[14] = lmat[14] - dti;
  mat[15] = lmat[15] - dti;
  mat[16] = lmat[16] - dti;
  mat[17] = lmat[17] - dti;
  mat[18] = lmat[18] - dti;
  mat[19] = lmat[19] - dti;
  mat[20] = lmat[20] - dti;
  mat[21] = lmat[21] - dti;
  mat[22] = lmat[22] - dti;
  mat[23] = lmat[23] - dti;
  mat[24] = lmat[24] - dti;
  mat[25] = lmat[25] - dti;
  mat[26] = lmat[26] - dti;
  mat[27] = lmat[27] - dti;
  mat[28] = lmat[28] - dti;
  mat[29] = lmat[29] - dti;
  mat[30] = lmat[30] - dti;
  mat[31] = lmat[31] - dti;
} // nlnmat

// initialize the solver (error tolerance)
KOKKOS_INLINE_FUNCTION
void imp_slv_inti(Real epsilon[clscnt4]) {
  for (int i = 0; i < clscnt4; ++i) {
    epsilon[i] = rel_err;
  }
}
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void newton_raphson_iter(
    const Real dti, const Real lin_jac[nzcnt], const Real lrxt[rxntot],
    const Real lhet[gas_pcnst],         // in
    const Real iter_invariant[clscnt4], // in
    const bool factor[itermax], VectorType &lsol,
    Real solution[clscnt4],                     // inout
    bool converged[clscnt4], bool &convergence, // out
    Real prod[clscnt4], Real loss[clscnt4], Real max_delta[clscnt4],
    // work array
    Real epsilon[clscnt4]) {

  constexpr auto clsmap_4 = gas_chemistry::clsmap_4;
  constexpr auto permute_4 = gas_chemistry::permute_4;
  // dti := 1 / dt
  // lrxt := reaction rates in 1D array [1/cm^3/s]
  // lhet := washout rates [1/s]
  // iter_invariant := dti * solution + ind_prd
  // factor := boolean controlling whether to do LU factorization
  // lsol is local solution--appears to be identical to 'solution'
  // looks like 'solution' is local to imp_sol() and 'lsol' is local to
  // newton_raphson_iter(), and the final solutions is 'base_sol'
  // these are volume mixing ratios [kmol species/kmol dry air]
  // lsol := base_sol (at initialization), then when converged base_sol = lsol
  // solution := array from imp_sol that holds the intermediate solutions and
  //         also holds the solution after converged [kmol species/kmol dry air]
  // converged := array for entrywise convergence bools
  // convergence := overall bool flag for convergence
  // prod/loss := chemical production/loss rates [1/cm^3/s]
  // NOTE: max_delta doesn't appear to be used for anything within gas_chem.hpp
  //       however, it looks like it's written to output in MAM4
  // max_delta := abs(forcing / solution) if abs(solution) > 1.0e-20 and
  //              0 otherwise
  // epsilon := rel_err = 1.0e-3 (hardcoded above)

  // -----------------------------------------------------
  //  the newton-raphson iteration for f(y) = 0
  // -----------------------------------------------------

  Real sys_jac[nzcnt] = {};
  Real forcing[clscnt4] = {};
  // BAD CONSTANT
  const Real small = 1.0e-40;
  const Real zero = 0;

  for (int nr_iter = 0; nr_iter < itermax; ++nr_iter) {
    // -----------------------------------------------------------------------
    //  ... the non-linear component
    // -----------------------------------------------------------------------

    if (factor[nr_iter]) {
      nlnmat(sys_jac, // out
             lin_jac,
             dti); // in
      // -----------------------------------------------------------------------
      //  ... factor the "system" matrix
      // -----------------------------------------------------------------------

      lu_fac(sys_jac);

    } // factor
    // -----------------------------------------------------------------------
    //  ... form f(y)
    // -----------------------------------------------------------------------
    imp_prod_loss(prod, loss,        // out
                  lsol, lrxt, lhet); // in

    // the units are internally consistent here, providing that
    // iter_invariant, prod, loss all have units [1/s] to match up with
    // solution (vmr) [-] and dti [1/s]. however, there could be other answers
    for (int mm = 0; mm < clscnt4; ++mm) {
      forcing[mm] =
          solution[mm] * dti - (iter_invariant[mm] + prod[mm] - loss[mm]);
    } // mm

    // -----------------------------------------------------------------------
    //  ... solve for the mixing ratio at t(n+1)
    // -----------------------------------------------------------------------
    lu_slv(sys_jac, forcing);
    for (int mm = 0; mm < clscnt4; ++mm) {
      solution[mm] += forcing[mm];
    } // mm

    // -----------------------------------------------------------------------
    //  ... convergence measures
    // -----------------------------------------------------------------------

    // NOTE: is there a particular reason we don't check on the first iteration?
    // seems like it'd be better to avoid the if on every iteration loop.
    // same deal below
    if (nr_iter > 0) {
      for (int kk = 0; kk < clscnt4; ++kk) {
        int mm = permute_4[kk];
        // BAD CONSTANT
        if (mam4::abs(solution[mm]) > 1.0e-20) {
          max_delta[kk] = mam4::abs(forcing[mm] / solution[mm]);
        } else {
          max_delta[kk] = zero;
        }

      } // kk

    } // nr_iter

    // -----------------------------------------------------------------------
    //  ... limit iterate
    // -----------------------------------------------------------------------
    for (int kk = 0; kk < clscnt4; ++kk) {
      if (solution[kk] < zero) {
        solution[kk] = zero;
      }
    } // end kk

    // -----------------------------------------------------------------------
    //  ... transfer latest solution back to work array
    // -----------------------------------------------------------------------

    for (int kk = 0; kk < clscnt4; ++kk) {
      int jj = clsmap_4[kk];
      int mm = permute_4[kk];
      lsol[jj] = solution[mm];
    } // end kk

    // -----------------------------------------------------------------------
    //  ... check for convergence
    // -----------------------------------------------------------------------

    if (nr_iter > 0) {
      convergence = true;
      for (int kk = 0; kk < clscnt4; ++kk) {
        converged[kk] = true;

        int mm = permute_4[kk];
        // TODO: is there a computational reason this needs to happen?
        // I suspect not, given that epsilon is hard-coded to 1e-3, meaning that
        // all of this logic surrounding 'converged[kk] = ...' is unnecessary
        bool frc_mask = mam4::abs(forcing[mm]) > small;
        if (frc_mask) {
          // this ends up effectively being:
          //                         if (small < abs(forcing) <= eps * abs(sol))
          //                            => converged
          // so the lower bound appears unnecessary
          converged[kk] =
              mam4::abs(forcing[mm]) <= epsilon[kk] * mam4::abs(solution[mm]);
        } else {
          // and this is just; if (abs(forcing) <= small <= eps) => converged
          // and the implicit comparison of small and eps is not helpful
          converged[kk] = true;
        } // frc_mask
        if (!converged[kk]) {
          convergence = false;
        }
      } // end

      if (convergence) {
        return;
      }
    } // end if (nr_iter > 0)
  }   // end nr_iter loop
} // newton_raphson_iter() function
template <typename VectorType>
KOKKOS_INLINE_FUNCTION ImpSolResult
imp_sol(VectorType &base_sol, // inout - species mixing ratios [vmr]
        const Real reaction_rates[rxntot], const Real het_rates[gas_pcnst],
        const Real extfrc[extcnt], const Real &delt, const bool factor[itermax],
        Real epsilon[clscnt4], Real prod_out[clscnt4], Real loss_out[clscnt4]) {

  ImpSolResult result;

  constexpr auto clsmap_4 = gas_chemistry::clsmap_4;
  constexpr auto permute_4 = gas_chemistry::permute_4;

  // ---------------------------------------------------------------------------
  //  ... imp_sol advances the volumetric mixing ratio
  //  forward one time step via the fully implicit euler scheme.
  //
  // NOTE: does anyone know what this is referring to?
  // can probably lose it since it looks like these chips were axed in 2020/2023
  // this source is meant for small l1 cache machines such as
  // the intel pentium and itanium cpus
  // ---------------------------------------------------------------------------

  // NOTE:
  // extfrc := external in-situ forcing [1/cm^3/s]

  const Real zero = 0;
  const Real half = 0.5;
  const Real one = 1;
  const Real two = 2;

  const int cut_limit = 5;

  result.outcome = ImpSolOutcome::InvalidInput;
  result.failed_attempts = 0;
  result.cut_count = 0;
  result.accepted_steps = 0;
  result.accepted_interval = 0;
  result.requested_interval = delt;
  for (int kk = 0; kk < clscnt4; ++kk) {
    prod_out[kk] = zero;
    loss_out[kk] = zero;
  }

  bool input_is_finite = Kokkos::isfinite(delt) && delt > zero;
  for (int mm = 0; mm < gas_pcnst; ++mm) {
    input_is_finite = input_is_finite && Kokkos::isfinite(base_sol[mm]) &&
                      Kokkos::isfinite(het_rates[mm]);
  }
  for (int mm = 0; mm < rxntot; ++mm) {
    input_is_finite = input_is_finite && Kokkos::isfinite(reaction_rates[mm]);
  }
  for (int mm = 0; mm < extcnt; ++mm) {
    input_is_finite = input_is_finite && Kokkos::isfinite(extfrc[mm]);
  }
  for (int kk = 0; kk < clscnt4; ++kk) {
    input_is_finite =
        input_is_finite && Kokkos::isfinite(epsilon[kk]) && epsilon[kk] >= zero;
  }
  if (!input_is_finite) {
    result.outcome = ImpSolOutcome::InvalidInput;
    return result;
  }

  Real ind_prd[clscnt4] = {};
  Real lin_jac[nzcnt] = {};
  bool converged[clscnt4] = {};
  bool convergence = false;
  Real prod[clscnt4] = {};
  Real loss[clscnt4] = {};
  Real max_delta[clscnt4] = {};

  // -----------------------------------------------------------------------
  //  ... class independent forcing
  // -----------------------------------------------------------------------
  // FIXME: BAD CONSTANT
  // what does this 4 represent, and would it ever be different?
  indprd(4,                       // in
         ind_prd,                 // inout
         reaction_rates, extfrc); // in

  Real solution[clscnt4] = {};
  Real iter_invariant[clscnt4] = {};
  Real lsol[gas_pcnst] = {};

  // !-----------------------------------------------------------------------
  //       ! ... time step loop
  //       !-----------------------------------------------------------------------
  Real dt = delt;
  int cut_cnt = 0;
  int stp_con_cnt = 0;
  // track how much of the outer time step = delt (interval) has been completed
  // during Newton-Raphson iteration
  Real interval_done = zero;
  // time_step_loop
  for (int i = 0; i < max_time_steps; ++i) {
    const Real attempted_dt = dt;
    const Real dti = one / dt;
    // -----------------------------------------------------------------------
    //  ... transfer from base to local work arrays
    // -----------------------------------------------------------------------
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      lsol[mm] = base_sol[mm];
    }
    convergence = false;
    for (int kk = 0; kk < clscnt4; ++kk) {
      converged[kk] = false;
      max_delta[kk] = zero;
    }
    // -----------------------------------------------------------------------
    //  ... transfer from base to class array
    // -----------------------------------------------------------------------

    for (int kk = 0; kk < clscnt4; ++kk) {
      int jj = clsmap_4[kk];
      int mm = permute_4[kk];
      solution[mm] = lsol[jj];
    } // kk

    // -----------------------------------------------------------------------
    //  ... set the iteration invariant part of the function f(y)
    // -----------------------------------------------------------------------

    // TODO: the units seem wrong here--could these arrays hold quantities
    // with different units?
    // ind_prd has units [1/cm^3/s] (for the entries that are nonzero)
    // dti units are [1/s], and
    // solution is a volume mixing ratio [kmol species/kmol dry air]
    // NOTE: this could be correct if solution had units [1/cm^3]
    // which would line up with a number concentration
    for (int mm = 0; mm < clscnt4; ++mm) {
      iter_invariant[mm] = dti * solution[mm] + ind_prd[mm];
    } // mm
    //-----------------------------------------------------------------------
    // ... the linear component
    //-----------------------------------------------------------------------
    linmat(lin_jac,                    //  out
           reaction_rates, het_rates); // in

    // =======================================================================
    //  the newton-raphson iteration for f(y) = 0
    // =======================================================================

    newton_raphson_iter(dti, lin_jac, reaction_rates, het_rates, // in
                        iter_invariant,                          // in
                        factor, lsol,
                        solution,                        // inout
                        converged, convergence,          // out
                        prod, loss, max_delta, epsilon); // out

    bool trial_is_finite = true;
    for (int mm = 0; mm < gas_pcnst; ++mm) {
      trial_is_finite = trial_is_finite && Kokkos::isfinite(lsol[mm]);
    }
    for (int kk = 0; kk < clscnt4; ++kk) {
      trial_is_finite = trial_is_finite && Kokkos::isfinite(solution[kk]) &&
                        Kokkos::isfinite(prod[kk]) &&
                        Kokkos::isfinite(loss[kk]);
    }

    if (!trial_is_finite) {
      result.failed_attempts += 1;
      result.outcome = ImpSolOutcome::NonfiniteIterate;
      return result;
    }

    // -----------------------------------------------------------------------
    //  ... check for newton-raphson convergence
    // -----------------------------------------------------------------------
    if (!convergence) {
      // -----------------------------------------------------------------------
      //            ... non-convergence
      // -----------------------------------------------------------------------
      result.failed_attempts += 1;
      stp_con_cnt = 0;

      if (cut_cnt < cut_limit) {
        cut_cnt += 1;
        result.cut_count = cut_cnt;
        if (cut_cnt < cut_limit) {
          dt *= half;
        } else {
          dt *= 0.1;
        } // cut_cnt < cut_limit
        // Retry this same physical interval from the last accepted state.
        continue;
      } else {
        result.outcome = ImpSolOutcome::CutLimitExhausted;
        return result;
      } //  cut_cnt < cut_limit
    }   // non-convergence

    // -----------------------------------------------------------------------
    // ... commit the converged trial and check for interval done
    // -----------------------------------------------------------------------

    for (int mm = 0; mm < gas_pcnst; ++mm) {
      base_sol[mm] = lsol[mm];
    }
    result.accepted_steps += 1;
    interval_done += attempted_dt;
    result.accepted_interval = interval_done;

    // BAD CONSTANT
    if (mam4::abs(delt - interval_done) <= 0.0001) {
      for (int kk = 0; kk < clscnt4; ++kk) {
        const int mm = permute_4[kk];
        prod_out[kk] = prod[mm] + ind_prd[mm];
        loss_out[kk] = loss[mm];
      }
      result.outcome = result.failed_attempts == 0
                           ? ImpSolOutcome::Converged
                           : ImpSolOutcome::ConvergedAfterRetry;
      return result;
    } else {
      // -----------------------------------------------------------------------
      //  ... transfer latest solution back to base array
      // -----------------------------------------------------------------------
      if (convergence) {
        stp_con_cnt += 1;
      }

      if (stp_con_cnt >= 2) {
        dt *= two;
        stp_con_cnt = 0;
      }

      dt = mam4::min(dt, delt - interval_done);

    } // abs( delt - interval_done ) <= .0001
  }   // time_step_loop

  result.outcome = ImpSolOutcome::MaximumStepsExhausted;
  return result;
} // imp_sol

} // namespace legacy

} // namespace gas_chemistry
} // namespace mam4

#include "gas_chem_direct.hpp"

#endif
