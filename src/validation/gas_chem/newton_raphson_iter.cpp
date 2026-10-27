// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/gas_chem.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::gas_chemistry;
using View1DHost = typename mam4::HostType::view_1d<Real>;

void newton_raphson_iter(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const auto lin_jac = input.get_array("lin_jac");
    const auto lrxt = input.get_array("lrxt");
    const auto lhet = input.get_array("lhet");
    const auto iter_invariant = input.get_array("iter_invariant");
    auto lsol = input.get_array("lsol");
    auto solution = input.get_array("solution");
    const auto dti = input.get_array("dti")[0];

    std::vector<Real> prod(clscnt4, zero);
    std::vector<Real> loss(clscnt4, zero);
    std::vector<Real> max_delta(clscnt4, zero);

    Real epsilon[clscnt4] = {};
    legacy::imp_slv_inti(epsilon);

    const bool factor[legacy::itermax] = {true};
    bool converged[clscnt4] = {true};
    bool convergence = false;

    View1DHost lsol_host("lsol", lsol.size());
    for (std::size_t i = 0; i < lsol.size(); ++i)
      lsol_host(i) = lsol[i];

    legacy::newton_raphson_iter(
        dti, lin_jac.data(), lrxt.data(), lhet.data(), iter_invariant.data(),
        factor, lsol_host, solution.data(), converged, convergence, prod.data(),
        loss.data(), max_delta.data(), epsilon);

    std::vector<Real> converged_out;
    for (int i = 0; i < clscnt4; ++i) {
      if (converged[i]) {
        converged_out.push_back(1);
      } else {
        converged_out.push_back(0);
      }
    }

    std::vector<Real> convergence_out;

    if (convergence) {
      convergence_out.push_back(1);
    } else {
      convergence_out.push_back(0);
    }

    for (std::size_t i = 0; i < lsol.size(); ++i)
      lsol[i] = lsol_host(i);
    output.set("lsol", lsol);
    output.set("solution", solution);
    output.set("converged", converged_out);
    output.set("convergence", convergence_out);
    output.set("prod", prod);
    output.set("loss", loss);
    output.set("max_delta", max_delta);
  });
}
