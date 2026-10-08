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
  int non_converged_species_idx = -1;
  int non_converged_species_count = 0;

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
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
imp_sol(VectorType &base_sol, // inout - species mixing ratios [vmr]
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

  result.outcome = ImpSolOutcome::InvalidInput;
  result.failed_attempts = 0;
  result.cut_count = 0;
  result.accepted_steps = 0;
  result.accepted_interval = 0;
  result.requested_interval = delt;
  result.non_converged_species_idx = -1;
  result.non_converged_species_count = 0;
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
      int non_conv_cnt = 0;
      for (int kk = 0; kk < clscnt4; ++kk) {
        if (!Kokkos::isfinite(solution[kk]) || !Kokkos::isfinite(prod[kk]) ||
            !Kokkos::isfinite(loss[kk])) {
          if (non_conv_cnt == 0) {
            result.non_converged_species_idx = clsmap_4[kk];
          }
          non_conv_cnt++;
        }
      }
      result.non_converged_species_count = non_conv_cnt;
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

      int non_conv_cnt = 0;
      for (int kk = 0; kk < clscnt4; ++kk) {
        if (!converged[kk]) {
          if (non_conv_cnt == 0) {
            result.non_converged_species_idx = clsmap_4[kk];
          }
          non_conv_cnt++;
        }
      }
      result.non_converged_species_count = non_conv_cnt;

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
    result.non_converged_species_idx = -1;
    result.non_converged_species_count = 0;

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
  int non_conv_cnt = 0;
  for (int kk = 0; kk < clscnt4; ++kk) {
    if (!converged[kk]) {
      if (non_conv_cnt == 0) {
        result.non_converged_species_idx = clsmap_4[kk];
      }
      non_conv_cnt++;
    }
  }
  result.non_converged_species_count = non_conv_cnt;
} // imp_sol

// -----------------------------------------------------------------------------
// Analytical integration helpers
// -----------------------------------------------------------------------------

// Helper for \int_0^dt exp(-lambda * t) dt = (1 - exp(-lambda * dt)) / lambda
KOKKOS_INLINE_FUNCTION
Real exp_decay_int(const Real lambda, const Real dt) {
  const Real x = lambda * dt;
  if (mam4::abs(x) < 1.0e-6) {
    return dt * (1.0 - 0.5 * x + (1.0 / 6.0) * x * x);
  }
  return (1.0 - mam4::exp(-x)) / lambda;
}

// Helper for \int_0^dt exp(-lambda_a * (dt - t)) * exp(-lambda_b * t) dt
//            = (exp(-lambda_b * dt) - exp(-lambda_a * dt)) / (lambda_a - lambda_b)
KOKKOS_INLINE_FUNCTION
Real exp_diff_int(const Real lambda_a, const Real lambda_b, const Real dt) {
  const Real d = (lambda_a - lambda_b) * dt;
  if (mam4::abs(d) < 1.0e-6) {
    const Real avg_lambda = 0.5 * (lambda_a + lambda_b);
    return dt * mam4::exp(-avg_lambda * dt) * (1.0 + (1.0 / 24.0) * d * d);
  }
  return (mam4::exp(-lambda_b * dt) - mam4::exp(-lambda_a * dt)) / (lambda_a - lambda_b);
}

// Helper for \int_0^dt exp(-lambda_a * (dt - t)) * (1 - exp(-lambda_b * t)) / lambda_b dt
KOKKOS_INLINE_FUNCTION
Real exp_decay_chain_int(const Real lambda_a, const Real lambda_b, const Real dt) {
  const Real x = lambda_b * dt;
  if (mam4::abs(x) < 1.0e-6) {
    const Real i1_a = exp_decay_int(lambda_a, dt);
    if (mam4::abs(lambda_a * dt) < 1.0e-6) {
      return 0.5 * dt * dt;
    }
    return (dt - i1_a) / lambda_a;
  }
  const Real i1_a = exp_decay_int(lambda_a, dt);
  const Real i2 = exp_diff_int(lambda_a, lambda_b, dt);
  return (i1_a - i2) / lambda_b;
}

// -----------------------------------------------------------------------------
// analytical_sol: Exact analytical solution for linear chemical ODE system:
// dy/dt = A * y + b
// -----------------------------------------------------------------------------
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
analytical_sol(VectorType &base_sol, // inout - species mixing ratios [vmr]
               const Real reaction_rates[rxntot], const Real het_rates[gas_pcnst],
               const Real extfrc[extcnt], const Real &delt,
               Real prod_out[clscnt4], Real loss_out[clscnt4],
               ImpSolResult &result) {

  constexpr auto clsmap_4 = gas_chemistry::clsmap_4;
  constexpr auto permute_4 = gas_chemistry::permute_4;

  const Real zero = 0;

  result.outcome = ImpSolOutcome::InvalidInput;
  result.failed_attempts = 0;
  result.cut_count = 0;
  result.accepted_steps = 0;
  result.accepted_interval = 0;
  result.requested_interval = delt;
  result.non_converged_species_idx = -1;
  result.non_converged_species_count = 0;
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
  if (!input_is_finite) {
    result.outcome = ImpSolOutcome::InvalidInput;
    return;
  }

  // Independent forcing and rates
  Real ind_prd[clscnt4] = {};
  indprd(4, ind_prd, reaction_rates, extfrc);

  // Initial concentrations for Class 4 species
  // Index mapping in base_sol:
  // base_sol[1] = H2O2  (clsmap_4[0])
  // base_sol[2] = H2SO4 (clsmap_4[1])
  // base_sol[3] = SO2   (clsmap_4[2])
  // base_sol[4] = DMS   (clsmap_4[3])
  // base_sol[5] = SOAG  (clsmap_4[4])
  // base_sol[6..30] = Aerosols (clsmap_4[5..29])

  // 1. H2O2 (index 1 in base_sol)
  const Real lambda_h2o2 = het_rates[1] + reaction_rates[0] + reaction_rates[2];
  const Real b_h2o2 = ind_prd[0]; // rxt[1]
  const Real h2o2_0 = base_sol[1];
  const Real h2o2_end = h2o2_0 * mam4::exp(-lambda_h2o2 * delt) + b_h2o2 * exp_decay_int(lambda_h2o2, delt);

  // 2. Coupled sulfur network: DMS -> SO2 -> H2SO4
  // DMS (index 4 in base_sol)
  const Real lambda_dms = het_rates[4] + reaction_rates[4] + reaction_rates[5] + reaction_rates[6];
  const Real dms_0 = base_sol[4];
  const Real dms_end = dms_0 * mam4::exp(-lambda_dms * delt);

  // SO2 (index 3 in base_sol)
  const Real lambda_so2 = het_rates[3] + reaction_rates[3];
  const Real k_dms_to_so2 = reaction_rates[4] + 0.5 * reaction_rates[5] + reaction_rates[6];
  const Real b_so2 = ind_prd[2]; // extfrc[0]
  const Real so2_0 = base_sol[3];
  const Real so2_end = so2_0 * mam4::exp(-lambda_so2 * delt)
                     + b_so2 * exp_decay_int(lambda_so2, delt)
                     + k_dms_to_so2 * dms_0 * exp_diff_int(lambda_so2, lambda_dms, delt);

  // H2SO4 (index 2 in base_sol)
  const Real lambda_h2so4 = het_rates[2];
  const Real k_so2_to_h2so4 = reaction_rates[3];
  const Real h2so4_0 = base_sol[2];

  // Integration of d(H2SO4)/dt = -lambda_h2so4 * H2SO4 + k_so2_to_h2so4 * SO2(t)
  const Real term_so2_init = so2_0 * exp_diff_int(lambda_h2so4, lambda_so2, delt);
  const Real term_so2_source = b_so2 * exp_decay_chain_int(lambda_h2so4, lambda_so2, delt);

  Real term_dms_chain = zero;
  const Real d_so2_dms = (lambda_so2 - lambda_dms) * delt;
  if (mam4::abs(d_so2_dms) < 1.0e-6) {
    const Real avg_lambda = 0.5 * (lambda_so2 + lambda_dms);
    term_dms_chain = delt * exp_diff_int(lambda_h2so4, avg_lambda, delt);
  } else {
    const Real i2_h2so4_dms = exp_diff_int(lambda_h2so4, lambda_dms, delt);
    const Real i2_h2so4_so2 = exp_diff_int(lambda_h2so4, lambda_so2, delt);
    term_dms_chain = (i2_h2so4_dms - i2_h2so4_so2) / (lambda_so2 - lambda_dms);
  }

  const Real h2so4_end = h2so4_0 * mam4::exp(-lambda_h2so4 * delt)
                       + k_so2_to_h2so4 * (term_so2_init + term_so2_source + k_dms_to_so2 * dms_0 * term_dms_chain);

  // Update base_sol for sulfur and H2O2
  base_sol[1] = mam4::max(zero, h2o2_end);
  base_sol[2] = mam4::max(zero, h2so4_end);
  base_sol[3] = mam4::max(zero, so2_end);
  base_sol[4] = mam4::max(zero, dms_end);

  // 3. Decoupled SOAG and aerosol species (Class 4 indices 4..29, base_sol indices 5..30)
  for (int kk = 4; kk < clscnt4; ++kk) {
    const int spc_idx = clsmap_4[kk];
    const Real lambda_spc = het_rates[spc_idx];
    const Real b_spc = ind_prd[kk];
    const Real spc_0 = base_sol[spc_idx];
    const Real spc_end = spc_0 * mam4::exp(-lambda_spc * delt) + b_spc * exp_decay_int(lambda_spc, delt);
    base_sol[spc_idx] = mam4::max(zero, spc_end);
  }

  // Compute final production and loss diagnostic outputs
  Real prod[clscnt4] = {};
  Real loss[clscnt4] = {};
  imp_prod_loss(prod, loss, base_sol, reaction_rates, het_rates);
  for (int kk = 0; kk < clscnt4; ++kk) {
    const int mm = permute_4[kk];
    prod_out[kk] = prod[mm] + ind_prd[mm];
    loss_out[kk] = loss[mm];
  }

  result.outcome = ImpSolOutcome::Converged;
  result.failed_attempts = 0;
  result.cut_count = 0;
  result.accepted_steps = 1;
  result.accepted_interval = delt;
  result.non_converged_species_idx = -1;
  result.non_converged_species_count = 0;
}


} // namespace gas_chemistry
} // namespace mam4
#endif
