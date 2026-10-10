// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void loadaer(Ensemble *ensemble) {
  using View1DHost = typename mam4::HostType::view_1d<Real>;
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const int ntot_amode = mam4::AeroConfig::num_modes();
    const int maxd_aspectype = mam4::ndrop::maxd_aspectype;

    auto state_q = input.get_array("state_q");
    const Real air_density = input.get_array("cs")[0];
    const Real phase = input.get_array("phase")[0];

    Real qcldbrn[maxd_aspectype][ntot_amode] = {};

    if (input.has_array("qcldbrn1d")) {
      const auto qcldbrn1d_1d = input.get_array("qcldbrn1d");
      int count = 0;
      for (int i = 0; i < ntot_amode; ++i) {
        for (int j = 0; j < maxd_aspectype; ++j) {
          qcldbrn[j][i] = qcldbrn1d_1d[count];
          count++;
        }
      }
    }

    std::vector<Real> qcldbrn1d_num;
    if (input.has_array("qcldbrn1d")) {
      qcldbrn1d_num = input.get_array("qcldbrn1d_num");
    } else {
      qcldbrn1d_num = {zero, zero, zero, zero};
    }
    std::vector<Real> naerosol(ntot_amode, zero), vaerosol(ntot_amode, zero),
        hygro(ntot_amode, zero);
    Real exp45logsig[mam4::AeroConfig::num_modes()],
        alogsig[mam4::AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[mam4::AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[mam4::AeroConfig::num_modes()] = {};

    Real aten = zero;

    mam4::ndrop::ndrop_init(exp45logsig, alogsig, aten,
                            num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                            num2vol_ratio_max_nmodes); // voltonumblo_amode
    View1DHost state(state_q.data(), mam4::aero_model::pcnst);
    mam4::ndrop::loadaer(state, air_density, phase, num2vol_ratio_min_nmodes,
                         num2vol_ratio_max_nmodes, qcldbrn,
                         qcldbrn1d_num.data(), naerosol.data(), vaerosol.data(),
                         hygro.data());

    output.set("naerosol", naerosol);
    output.set("vaerosol", vaerosol);
    output.set("hygro", hygro);
  });
}
