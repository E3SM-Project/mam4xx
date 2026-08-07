// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void ccncalc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // number of vertical points.
    // validation test from standalone ndrop.
    const Real zero = 0;
    const int maxd_aspectype = mam4::ndrop::maxd_aspectype;
    const int ntot_amode = mam4::AeroConfig::num_modes();
    const int pcnst = mam4::aero_model::pcnst;
    const int psat = mam4::ndrop::psat;

    const int pver = mam4::ndrop::pver;
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");

    const int top_lev = 6;

    using View2D = mam4::ndrop::View2D;
    using View1DHost = typename mam4::HostType::view_1d<Real>;

    View2D state_q("state_q", pcnst, pver);
    auto state_host = Kokkos::create_mirror_view(state_q);

    int count = 0;
    for (int i = 0; i < pcnst; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
        state_host(i, kk) = state_q_db[count];
        count++;
      }
    }

    Kokkos::deep_copy(state_q, state_host);

    mam4::ColumnView tair;
    mam4::ColumnView pmid;
    tair = mam4::testing::create_column_view(pver);
    pmid = mam4::testing::create_column_view(pver);

    auto tair_host = View1DHost((Real *)tair_db.data(), pver);
    auto pmid_host = View1DHost((Real *)pmid_db.data(), pver);

    Kokkos::deep_copy(tair, tair_host);
    Kokkos::deep_copy(pmid, pmid_host);

    View2D ccn("ccn", psat, pver);

    Kokkos::parallel_for(
        "ccncalc", pver - top_lev, KOKKOS_LAMBDA(int k) {
          Real qcldbrn[maxd_aspectype][ntot_amode] = {{zero}};
          Real qcldbrn_num[ntot_amode] = {zero};

          const int kk = k + top_lev;
          const auto state_q_k = Kokkos::subview(state_q, Kokkos::ALL(), kk);
          const Real air_density =
              mam4::conversions::density_of_ideal_gas(tair(kk), pmid(kk));

          Real exp45logsig[mam4::AeroConfig::num_modes()],
              alogsig[mam4::AeroConfig::num_modes()],
              num2vol_ratio_min_nmodes[mam4::AeroConfig::num_modes()],
              num2vol_ratio_max_nmodes[mam4::AeroConfig::num_modes()] = {};

          Real aten = zero;

          mam4::ndrop::ndrop_init(
              exp45logsig, alogsig, aten,
              num2vol_ratio_min_nmodes,  // voltonumbhi_amode
              num2vol_ratio_max_nmodes); // voltonumblo_amode

          const auto ccn_k = Kokkos::subview(ccn, Kokkos::ALL(), kk);
          mam4::ndrop::ccncalc(state_q_k, tair(kk), qcldbrn, qcldbrn_num,
                               air_density, num2vol_ratio_min_nmodes,
                               num2vol_ratio_max_nmodes, exp45logsig, alogsig,
                               ccn_k);
        });

    auto ccn_host = Kokkos::create_mirror_view(ccn);
    Kokkos::deep_copy(ccn_host, ccn);
    for (int i = 0; i < psat; ++i) {
      std::vector<Real> ccn_v(pver);
      for (int kk = 0; kk < pver; ++kk) {
        ccn_v[kk] = ccn_host(i, kk);
      }
      output.set("ccn_" + std::to_string(i + 1), ccn_v);
    }
  });
}
