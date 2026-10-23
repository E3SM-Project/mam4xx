// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void maxsat(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zeta = input.get_array("zeta")[0];
    int nmode = input.get_array("nmode")[0];

    std::vector<Real> smc(nmode);
    std::vector<Real> eta(nmode);
    eta = input.get_array("eta");
    smc = input.get_array("smc");
    const Real smax = mam4::ndrop::maxsat(zeta, eta.data(), nmode, smc.data());
    output.set("smax", smax);
  });
}
