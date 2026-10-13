// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::gas_chemistry;

void imp_prod_loss(Ensemble *ensemble) {

  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    std::vector<Real> prod(clscnt4, zero);
    std::vector<Real> loss(clscnt4, zero);

    const auto rxt = input.get_array("rxt");
    const auto het_rates = input.get_array("het_rates");
    auto y = input.get_array("y");

    imp_prod_loss(prod.data(), loss.data(), y.data(), rxt.data(),
                  het_rates.data());
    output.set("prod", prod);
    output.set("loss", loss);
  });
}
