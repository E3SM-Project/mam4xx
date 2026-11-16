// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::modal_aero_opt;

void binterp(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    constexpr Real zero = 0;
    using View1D = mam4::DeviceType::view_1d<Real>;
    using View1DHost = typename mam4::HostType::view_1d<Real>;

    const auto table_db = input.get_array("table");
    const auto ref_real = input.get_array("ref_real")[0];
    const auto ref_img = input.get_array("ref_img")[0];
    const auto ref_real_tab_db = input.get_array("ref_real_tab");
    const auto ref_img_tab_db = input.get_array("ref_img_tab");

    auto ref_real_tab_host =
        View1DHost((Real *)ref_real_tab_db.data(), ref_real_tab_db.size());
    View1D ref_real_tab("ref_real_tab", ref_real_tab_db.size());
    Kokkos::deep_copy(ref_real_tab, ref_real_tab_host);

    auto ref_img_tab_host =
        View1DHost((Real *)ref_img_tab_db.data(), ref_img_tab_db.size());
    View1D ref_img_tab("ref_img_tab", ref_img_tab_db.size());
    Kokkos::deep_copy(ref_img_tab, ref_img_tab_host);

    int itab_1 = 0;
    int ncoef = 5;
    int prefr = 7;
    int prefi = 10;

    View3D table("table", ncoef, prefr, prefi);
    auto table_host = Kokkos::create_mirror_view(table);

    int N1 = ncoef;
    int N2 = prefr;
    int N3 = prefi;

    for (int d1 = 0; d1 < N1; ++d1)
      for (int d2 = 0; d2 < N2; ++d2)
        for (int d3 = 0; d3 < N3; ++d3) {
          const int offset = d1 + N1 * (d2 + d3 * N2);
          table_host(d1, d2, d3) = table_db[offset];

        } // d3

    Kokkos::deep_copy(table, table_host);

    View1D coef("coef", ncoef);
    View1D tab("tab", 4);

    auto team_policy = mam4::ThreadTeamPolicy(1u, mam4::testing::team_size);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          int itab = 0;
          int jtab = 0;
          Real ttab = zero;
          Real utab = zero;
          binterp(table, ref_real, ref_img, ref_real_tab, ref_img_tab, itab,
                  jtab, ttab, utab, coef.data(), itab_1);
          tab(0) = Real(itab);
          tab(1) = Real(jtab);
          tab(2) = ttab;
          tab(3) = utab;
        });

    auto tab_host = Kokkos::create_mirror_view(tab);
    Kokkos::deep_copy(tab_host, tab);

    auto coef_host = Kokkos::create_mirror_view(coef);
    Kokkos::deep_copy(coef_host, coef);
    std::vector<Real> coef_out(coef_host.data(), coef_host.data() + ncoef);

    output.set("itab", std::vector<Real>(1, tab_host(0) + 1));
    output.set("jtab", std::vector<Real>(1, tab_host(1) + 1));
    output.set("ttab", std::vector<Real>(1, tab_host(2)));
    output.set("utab", std::vector<Real>(1, tab_host(3)));
    output.set("coef", coef_out);
  });
}
