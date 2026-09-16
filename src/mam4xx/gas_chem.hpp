#ifndef MAM4XX_GAS_CHEMISTRY_HPP
#define MAM4XX_GAS_CHEMISTRY_HPP

#include "gas_chem_mechanism.hpp"
#include "mam4_math.hpp"

#include <ekat_kernel_assert.hpp>

namespace mam4 {

namespace gas_chemistry {

// BAD CONSTANTs
constexpr int itermax = 11;
const Real rel_err = 1.0e-3;
// NOTE: high_rel_err is unused currently
// const Real high_rel_err = 1.0e-4;
const int max_time_steps = 1000;

enum class ImpSolOutcome {
  Converged,
  ConvergedAfterRetry,
  InvalidInput,
  NonfiniteIterate,
  CutLimitExhausted,
  MaximumStepsExhausted
};

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

KOKKOS_INLINE_FUNCTION
void usrrxt(Real rxt[rxntot], // inout
            const Real temperature, const Real invariants[nfs], const Real mtot,
            const int usr_HO2_HO2_ndx, const int usr_DMS_OH_ndx,
            const int usr_SO2_OH_ndx, const int inv_h2o_ndx) {

  /*-----------------------------------------------------------------
   ... ho2 + ho2 --> h2o2
   note: this rate involves the water vapor number density
  -----------------------------------------------------------------*/
  const Real one = 1.0;
  if (usr_HO2_HO2_ndx > 0) {
    // BAD CONSTANT
    const Real ko = 3.5e-13 * mam4::exp(430.0 / temperature);
    const Real kinf = 1.7e-33 * mtot * mam4::exp(1000. / temperature);
    const Real fc = one + 1.4e-21 * invariants[inv_h2o_ndx] *
                              mam4::exp(2200. / temperature);
    rxt[usr_HO2_HO2_ndx] = (ko + kinf) * fc;
  }

  /*-----------------------------------------------------------------
       ... DMS + OH  --> .5 * SO2
   -----------------------------------------------------------------*/
  if (usr_DMS_OH_ndx > 0) {
    // BAD CONSTANT
    const Real ko =
        one + 5.5e-31 * mam4::exp(7460. / temperature) * mtot * 0.21;
    rxt[usr_DMS_OH_ndx] =
        1.7e-42 * mam4::exp(7810. / temperature) * mtot * 0.21 / ko;
  }

  /*-----------------------------------------------------------------
         ... SO2 + OH  --> SO4  (REFERENCE?? - not Liao)
  -----------------------------------------------------------------*/
  if (usr_SO2_OH_ndx > 0) {
    // BAD CONSTANT
    const Real fc = 3.0e-31 * mam4::pow(300. / temperature, 3.3);
    const Real ko = fc * mtot / (one + fc * mtot / 1.5e-12);
    rxt[usr_SO2_OH_ndx] =
        ko *
        mam4::pow(0.6, one / (one + square(mam4::log10(fc * mtot / 1.5e-12))));
  }

} // usrrxt

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
namespace detail {
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void imp_sol_impl(
    VectorType &base_sol, // inout - species mixing ratios [vmr]
    const Real reaction_rates[rxntot], const Real het_rates[gas_pcnst],
    const Real extfrc[extcnt], const Real &delt, const bool factor[itermax],
    Real epsilon[clscnt4], Real prod_out[clscnt4], Real loss_out[clscnt4],
    ImpSolResult &result) {

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

  result = ImpSolResult{};
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
    input_is_finite =
        input_is_finite && Kokkos::isfinite(reaction_rates[mm]);
  }
  for (int mm = 0; mm < extcnt; ++mm) {
    input_is_finite = input_is_finite && Kokkos::isfinite(extfrc[mm]);
  }
  for (int kk = 0; kk < clscnt4; ++kk) {
    input_is_finite = input_is_finite && Kokkos::isfinite(epsilon[kk]) &&
                      epsilon[kk] >= zero;
  }
  if (!input_is_finite) {
    result.outcome = ImpSolOutcome::InvalidInput;
    return;
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
      return;
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
        return;
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
      return;
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
} // imp_sol_impl
} // namespace detail

template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
imp_sol(VectorType &base_sol, // inout - species mixing ratios [vmr]
        const Real reaction_rates[rxntot], const Real het_rates[gas_pcnst],
        const Real extfrc[extcnt], const Real &delt, const bool factor[itermax],
        Real epsilon[clscnt4], Real prod_out[clscnt4], Real loss_out[clscnt4],
        ImpSolResult &result) {
  detail::imp_sol_impl(base_sol, reaction_rates, het_rates, extfrc, delt,
                       factor, epsilon, prod_out, loss_out, result);
}

// Backward-compatible production entry point. A terminal solver outcome must
// never be silently consumed by aqueous chemistry or aerosol microphysics.
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
imp_sol(VectorType &base_sol, // inout - species mixing ratios [vmr]
        const Real reaction_rates[rxntot], const Real het_rates[gas_pcnst],
        const Real extfrc[extcnt], const Real &delt, const bool factor[itermax],
        Real epsilon[clscnt4], Real prod_out[clscnt4], Real loss_out[clscnt4]) {
  ImpSolResult result;
  imp_sol(base_sol, reaction_rates, het_rates, extfrc, delt, factor, epsilon,
          prod_out, loss_out, result);
  if (!result.success()) {
    EKAT_KERNEL_ERROR_MSG(
        "ERROR: imp_sol did not complete the requested chemistry interval.\n");
  }
}
} // namespace gas_chemistry
} // namespace mam4
#endif
