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
using View1DHost = typename mam4::HostType::view_1d<Real>;

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

    View1DHost base_sol_host("base_sol", base_sol.size());
    for (std::size_t i = 0; i < base_sol.size(); ++i)
      base_sol_host(i) = base_sol[i];

    // This Fortran-baseline comparison always exercises the legacy solver.
    Real epsilon[clscnt4] = {};
    legacy::imp_slv_inti(epsilon);
    bool factor[legacy::itermax];
    for (int i = 0; i < legacy::itermax; ++i) {
      factor[i] = true;
    }
    const auto result = legacy::imp_sol(
        base_sol_host, reaction_rates.data(), het_rates.data(), extfrc.data(),
        delt, factor, epsilon, prod_out.data(), loss_out.data());
    if (!result.success()) {
      std::cerr << "Legacy gas-chemistry validation failed (outcome code "
                << static_cast<int>(result.outcome) << ", accepted interval "
                << result.accepted_interval << " of "
                << result.requested_interval
                << " seconds). Stopping validation." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    for (std::size_t i = 0; i < base_sol.size(); ++i)
      base_sol[i] = base_sol_host(i);
    output.set("base_sol", base_sol);
  });
}
