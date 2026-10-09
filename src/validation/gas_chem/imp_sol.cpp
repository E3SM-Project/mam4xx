// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

#include <cstdlib>
#include <iostream>

using namespace skywalker;
using namespace mam4::gas_chemistry;

void imp_sol(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    auto base_sol = input.get_array("base_sol");
    const auto reaction_rates = input.get_array("reaction_rates");
    const auto het_rates = input.get_array("het_rates");
    const auto extfrc = input.get_array("extfrc");
    auto delt = input.get_array("delt")[0];

    std::vector<Real> prod_out(clscnt4, zero);
    std::vector<Real> loss_out(clscnt4, zero);

    ImpSolResult result;
    imp_sol(base_sol, reaction_rates.data(), het_rates.data(), extfrc.data(),
            delt, prod_out.data(), loss_out.data(), result);
    if (!result.success()) {
      const char *reason = "unknown solver outcome";
      switch (result.outcome) {
      case ImpSolOutcome::InvalidInput:
        reason = "invalid input: every input must be finite, the timestep "
                 "must be positive, and reaction rates and heterogeneous "
                 "losses for state entries 1-30 must be nonnegative";
        break;
      case ImpSolOutcome::NonfiniteResult:
        reason = "an intermediate calculation or final state/rate became "
                 "not-a-number (NaN) or infinity";
        break;
      case ImpSolOutcome::UnsafeDenominator:
        reason = "the backward-Euler denominator 1 - dt*diagonal was zero "
                 "or negative";
        break;
      case ImpSolOutcome::NegativeResult:
        reason = "the backward-Euler update produced a negative state entry";
        break;
      case ImpSolOutcome::Converged:
        break;
      }
      std::cerr << "Gas-chemistry validation failed: " << reason
                << " (outcome code " << static_cast<int>(result.outcome)
                << "). Chemical state was left unchanged and production/loss "
                   "outputs were set to zero. Stopping validation."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    output.set("base_sol", base_sol);
  });
}
