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
      std::cerr << "imp_sol validation did not complete the requested interval: "
                << static_cast<int>(result.outcome) << std::endl;
      std::exit(EXIT_FAILURE);
    }

    output.set("base_sol", base_sol);
  });
}
