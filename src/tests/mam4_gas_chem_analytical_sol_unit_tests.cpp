// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/floating_point.hpp>
#include <mam4xx/gas_chem.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <vector>

using mam4::Real;
using namespace mam4::gas_chemistry;

namespace {

void setup_test_conditions(std::vector<Real> &base_sol,
                           Real reaction_rates[rxntot],
                           Real het_rates[gas_pcnst],
                           Real extfrc[extcnt]) {
  base_sol.assign(gas_pcnst, 1.0e-15);

  // Set representative mixing ratios [vmr]
  base_sol[0] = 5.0e-8;  // O3
  base_sol[1] = 1.0e-9;  // H2O2
  base_sol[2] = 1.0e-12; // H2SO4
  base_sol[3] = 2.0e-10; // SO2
  base_sol[4] = 5.0e-11; // DMS
  base_sol[5] = 1.0e-11; // SOAG

  // Aerosol species
  for (int i = 6; i < gas_pcnst; ++i) {
    base_sol[i] = 1.0e-12 * (1.0 + 0.1 * (i - 6));
  }

  // Set representative reaction rates [1/s]
  reaction_rates[0] = 1.0e-5;  // J_H2O2
  reaction_rates[1] = 1.0e-16; // HO2 + HO2 -> H2O2
  reaction_rates[2] = 1.7e-6;  // OH + H2O2
  reaction_rates[3] = 1.0e-6;  // OH + SO2 -> H2SO4
  reaction_rates[4] = 5.0e-6;  // OH + DMS (abstraction)
  reaction_rates[5] = 2.0e-6;  // OH + DMS (addition)
  reaction_rates[6] = 1.0e-7;  // NO3 + DMS

  // Set heterogeneous loss/washout rates [1/s]
  for (int i = 0; i < gas_pcnst; ++i) {
    het_rates[i] = 1.0e-5;
  }
  het_rates[2] = 5.0e-4; // H2SO4 condensation sink is fast

  // External in-situ emissions [vmr/s]
  for (int i = 0; i < extcnt; ++i) {
    extfrc[i] = 0.0;
  }
  extfrc[0] = 1.0e-15; // SO2 surface/in-situ emissions
  extfrc[1] = 2.0e-16; // Aerosol emission
}

} // namespace

TEST_CASE("analytical_sol vs imp_sol: Small timestep convergence", "[gas_chem]") {
  std::vector<Real> base_sol_init;
  Real reaction_rates[rxntot] = {};
  Real het_rates[gas_pcnst] = {};
  Real extfrc[extcnt] = {};
  setup_test_conditions(base_sol_init, reaction_rates, het_rates, extfrc);

  // For a small time step (dt = 0.5 s), Backward Euler (imp_sol) and the exact
  // analytical solution should agree closely (relative error ~ O(dt)).
  const Real delt = 0.5;

  // 1. Solve with imp_sol
  std::vector<Real> sol_imp = base_sol_init;
  Real prod_imp[clscnt4] = {};
  Real loss_imp[clscnt4] = {};
  Real epsilon[clscnt4] = {};
  imp_slv_inti(epsilon);
  bool factor[itermax];
  for (int i = 0; i < itermax; ++i) {
    factor[i] = true;
  }
  ImpSolResult res_imp;
  imp_sol(sol_imp, reaction_rates, het_rates, extfrc, delt, factor, epsilon,
          prod_imp, loss_imp, res_imp);

  REQUIRE(res_imp.outcome == ImpSolOutcome::Converged);

  // 2. Solve with analytical_sol
  std::vector<Real> sol_ana = base_sol_init;
  Real prod_ana[clscnt4] = {};
  Real loss_ana[clscnt4] = {};
  ImpSolResult res_ana;
  analytical_sol(sol_ana, reaction_rates, het_rates, extfrc, delt,
                 prod_ana, loss_ana, res_ana);

  REQUIRE(res_ana.outcome == ImpSolOutcome::Converged);

  // 3. Compare key chemical species: H2O2 (1), H2SO4 (2), SO2 (3), DMS (4)
  for (int spc = 1; spc <= 4; ++spc) {
    const Real rel_diff =
        std::abs(sol_imp[spc] - sol_ana[spc]) / std::abs(sol_ana[spc]);
    // With dt = 0.5 s, relative difference is well below 0.1% (1e-3)
    REQUIRE(rel_diff < 1.0e-3);
  }

  // Aerosol species should also match closely
  for (int spc = 5; spc < gas_pcnst; ++spc) {
    const Real rel_diff =
        std::abs(sol_imp[spc] - sol_ana[spc]) / std::abs(sol_ana[spc]);
    REQUIRE(rel_diff < 1.0e-3);
  }
}

TEST_CASE("analytical_sol vs imp_sol: Atmospheric timestep (1800s)", "[gas_chem]") {
  std::vector<Real> base_sol_init;
  Real reaction_rates[rxntot] = {};
  Real het_rates[gas_pcnst] = {};
  Real extfrc[extcnt] = {};
  setup_test_conditions(base_sol_init, reaction_rates, het_rates, extfrc);

  const Real delt = 1800.0; // 30-minute climate model time step

  // 1. Solve with imp_sol
  std::vector<Real> sol_imp = base_sol_init;
  Real prod_imp[clscnt4] = {};
  Real loss_imp[clscnt4] = {};
  Real epsilon[clscnt4] = {};
  imp_slv_inti(epsilon);
  bool factor[itermax];
  for (int i = 0; i < itermax; ++i) {
    factor[i] = true;
  }
  ImpSolResult res_imp;
  imp_sol(sol_imp, reaction_rates, het_rates, extfrc, delt, factor, epsilon,
          prod_imp, loss_imp, res_imp);

  // 2. Solve with analytical_sol
  std::vector<Real> sol_ana = base_sol_init;
  Real prod_ana[clscnt4] = {};
  Real loss_ana[clscnt4] = {};
  ImpSolResult res_ana;
  analytical_sol(sol_ana, reaction_rates, het_rates, extfrc, delt,
                 prod_ana, loss_ana, res_ana);

  REQUIRE(res_ana.outcome == ImpSolOutcome::Converged);
  REQUIRE(res_ana.accepted_steps == 1);
  REQUIRE(res_ana.accepted_interval == delt);

  // Verify non-negativity across all species
  for (int i = 0; i < gas_pcnst; ++i) {
    REQUIRE(sol_ana[i] >= 0.0);
    REQUIRE(sol_imp[i] >= 0.0);
    REQUIRE(std::isfinite(sol_ana[i]));
    REQUIRE(std::isfinite(sol_imp[i]));
  }

  // Consistent qualitative physical behavior:
  // - DMS must be depleted by oxidation + deposition
  REQUIRE(sol_ana[4] < base_sol_init[4]);
  REQUIRE(sol_imp[4] < base_sol_init[4]);

  // - Production and loss diagnostics must be finite and non-negative
  for (int k = 0; k < clscnt4; ++k) {
    REQUIRE(prod_ana[k] >= 0.0);
    REQUIRE(loss_ana[k] >= 0.0);
    REQUIRE(std::isfinite(prod_ana[k]));
    REQUIRE(std::isfinite(loss_ana[k]));
  }
}

TEST_CASE("analytical_sol: Pure exponential decay exactness", "[gas_chem]") {
  std::vector<Real> base_sol(gas_pcnst, 1.0e-10);
  Real reaction_rates[rxntot] = {}; // Zero reaction rates
  Real het_rates[gas_pcnst] = {};
  Real extfrc[extcnt] = {}; // Zero external forcings

  const Real lambda = 2.0e-4; // 1/s
  for (int i = 0; i < gas_pcnst; ++i) {
    het_rates[i] = lambda;
  }

  const Real delt = 1000.0;
  Real prod_out[clscnt4] = {};
  Real loss_out[clscnt4] = {};
  ImpSolResult res;
  analytical_sol(base_sol, reaction_rates, het_rates, extfrc, delt,
                 prod_out, loss_out, res);

  REQUIRE(res.outcome == ImpSolOutcome::Converged);

  // For uncoupled pure decay, y(dt) = y0 * exp(-lambda * dt) exactly
  const Real expected = 1.0e-10 * std::exp(-lambda * delt);
  for (int i = 1; i < gas_pcnst; ++i) {
    const Real rel_err = std::abs(base_sol[i] - expected) / expected;
    REQUIRE(rel_err < 1.0e-12);
  }
}

TEST_CASE("analytical_sol: Degenerate loss rates in sulfur chain", "[gas_chem]") {
  std::vector<Real> base_sol;
  Real reaction_rates[rxntot] = {};
  Real het_rates[gas_pcnst] = {};
  Real extfrc[extcnt] = {};
  setup_test_conditions(base_sol, reaction_rates, het_rates, extfrc);

  // Set identical loss rates for SO2 and DMS to exercise the degenerate branch
  // (|lambda_SO2 - lambda_DMS| -> 0) in exp_diff_int without division by zero
  const Real equal_lambda = 1.0e-4;
  reaction_rates[3] = 0.0;
  reaction_rates[4] = 0.0;
  reaction_rates[5] = 0.0;
  reaction_rates[6] = 0.0;
  het_rates[3] = equal_lambda; // SO2
  het_rates[4] = equal_lambda; // DMS
  het_rates[2] = equal_lambda; // H2SO4

  const Real delt = 600.0;
  Real prod_out[clscnt4] = {};
  Real loss_out[clscnt4] = {};
  ImpSolResult res;
  analytical_sol(base_sol, reaction_rates, het_rates, extfrc, delt,
                 prod_out, loss_out, res);

  REQUIRE(res.outcome == ImpSolOutcome::Converged);
  REQUIRE(std::isfinite(base_sol[2])); // H2SO4
  REQUIRE(std::isfinite(base_sol[3])); // SO2
  REQUIRE(std::isfinite(base_sol[4])); // DMS
  REQUIRE(base_sol[2] >= 0.0);
  REQUIRE(base_sol[3] >= 0.0);
  REQUIRE(base_sol[4] >= 0.0);
}

TEST_CASE("analytical_sol: Invalid input handling", "[gas_chem]") {
  std::vector<Real> base_sol(gas_pcnst, 1.0e-10);
  Real reaction_rates[rxntot] = {};
  Real het_rates[gas_pcnst] = {};
  Real extfrc[extcnt] = {};
  Real prod_out[clscnt4] = {};
  Real loss_out[clscnt4] = {};
  ImpSolResult res;

  // Negative time step
  analytical_sol(base_sol, reaction_rates, het_rates, extfrc, -10.0,
                 prod_out, loss_out, res);
  REQUIRE(res.outcome == ImpSolOutcome::InvalidInput);

  // Non-finite initial concentration
  base_sol[1] = NAN;
  analytical_sol(base_sol, reaction_rates, het_rates, extfrc, 10.0,
                 prod_out, loss_out, res);
  REQUIRE(res.outcome == ImpSolOutcome::InvalidInput);
}
