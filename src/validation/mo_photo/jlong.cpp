// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::mo_photo;

void jlong(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename mam4::HostType::view_1d<Real>;
    using View1D = typename mam4::DeviceType::view_1d<Real>;
    const auto alb_in_db = input.get_array("alb_in");
    const Real sza_in = input.get_array("sza_in")[0];
    const auto p_in_db = input.get_array("p_in");
    const auto colo3_in_db = input.get_array("colo3_in");
    const auto t_in_db = input.get_array("t_in");

    auto alb_in_host = View1DHost((Real *)alb_in_db.data(), pver);
    auto p_in_host = View1DHost((Real *)p_in_db.data(), pver);
    auto colo3_in_host = View1DHost((Real *)colo3_in_db.data(), pver);
    auto t_in_host = View1DHost((Real *)t_in_db.data(), pver);

    const auto alb_in = View1D("alb_in", pver);
    const auto p_in = View1D("p_in", pver);
    const auto colo3_in = View1D("colo3_in", pver);
    const auto t_in = View1D("t_in", pver);

    Kokkos::deep_copy(alb_in, alb_in_host);
    Kokkos::deep_copy(p_in, p_in_host);
    Kokkos::deep_copy(colo3_in, colo3_in_host);
    Kokkos::deep_copy(t_in, t_in_host);

    const auto sza_db = input.get_array("sza");
    const auto del_sza_db = input.get_array("del_sza");
    const auto alb_db = input.get_array("alb");
    const auto press_db = input.get_array("press");
    const auto del_p_db = input.get_array("del_p");
    const auto colo3_db = input.get_array("colo3");
    const auto o3rat_db = input.get_array("o3rat");
    const auto del_alb_db = input.get_array("del_alb");
    const auto del_o3rat_db = input.get_array("del_o3rat");
    const auto etfphot_db = input.get_array("etfphot");

    auto shape_rsf_tab = input.get_array("shape_rsf_tab");
    auto synthetic_values = input.get_array("synthetic_values_rsf_tab");

    const int nw = std::min(int(etfphot_db.size()), int(shape_rsf_tab[0]));
    const int nump = std::min(int(press_db.size()), int(shape_rsf_tab[1]));
    const int numsza = std::min(int(sza_db.size()), int(shape_rsf_tab[2]));
    const int numcolo3 = std::min(int(o3rat_db.size()), int(shape_rsf_tab[3]));
    const int numalb = std::min(int(alb_db.size()), int(shape_rsf_tab[4]));

    View5D rsf_tab;

    mam4::validation::create_synthetic_rsf_tab(
        rsf_tab, nw, nump, numsza, numcolo3, numalb, synthetic_values.data());

    auto sza_host = View1DHost((Real *)sza_db.data(), numsza);
    const auto sza = View1D("sza", numsza);
    Kokkos::deep_copy(sza, sza_host);

    auto del_sza_host = View1DHost((Real *)del_sza_db.data(), numsza - 1);
    const auto del_sza = View1D("del_sza", numsza - 1);
    Kokkos::deep_copy(del_sza, del_sza_host);

    auto alb_host = View1DHost((Real *)alb_db.data(), numalb);
    const auto alb = View1D("alb", numalb);
    Kokkos::deep_copy(alb, alb_host);

    auto press_host = View1DHost((Real *)press_db.data(), nump);
    const auto press = View1D("press", nump);
    Kokkos::deep_copy(press, press_host);

    auto del_p_host = View1DHost((Real *)del_p_db.data(), nump - 1);
    const auto del_p = View1D("del_p", nump - 1);
    Kokkos::deep_copy(del_p, del_p_host);

    auto colo3_host = View1DHost((Real *)colo3_db.data(), nump);
    const auto colo3 = View1D("colo3", nump);
    Kokkos::deep_copy(colo3, colo3_host);

    auto o3rat_host = View1DHost((Real *)o3rat_db.data(), numcolo3);
    const auto o3rat = View1D("o3rat", numcolo3);
    Kokkos::deep_copy(o3rat, o3rat_host);

    auto del_alb_host = View1DHost((Real *)del_alb_db.data(), numalb - 1);
    const auto del_alb = View1D("del_alb", numalb - 1);
    Kokkos::deep_copy(del_alb, del_alb_host);

    auto del_o3rat_host = View1DHost((Real *)del_o3rat_db.data(), numcolo3 - 1);
    const auto del_o3rat = View1D("del_o3rat", numcolo3 - 1);
    Kokkos::deep_copy(del_o3rat, del_o3rat_host);

    auto etfphot_host = View1DHost((Real *)etfphot_db.data(), nw);
    const auto etfphot = View1D("etfphot", nw);
    Kokkos::deep_copy(etfphot, etfphot_host);

    // auto _host = View1DHost((Real *).data(), );
    // const auto  = View1D("", );
    // Kokkos::deep_copy(, );

    auto shape_xsqy = input.get_array("shape_xsqy");
    auto synthetic_values_xsqy = input.get_array("synthetic_values_xsqy");

    const int numj = int(shape_xsqy[0]);
    const int nt = int(shape_xsqy[2]);
    const int np_xs = int(shape_xsqy[3]);

    const auto prs_db = input.get_array("prs");
    const auto dprs_db = input.get_array("dprs");

    auto prs_host = View1DHost((Real *)prs_db.data(), np_xs);
    const auto prs = View1D("prs", np_xs);
    Kokkos::deep_copy(prs, prs_host);

    auto dprs_host = View1DHost((Real *)dprs_db.data(), np_xs - 1);
    const auto dprs = View1D("dprs", np_xs - 1);
    Kokkos::deep_copy(dprs, dprs_host);

    View2D rsf("rsf", nw, pver);
    View4D xsqy("xsqy", numj, nw, nt, np_xs);

    const Real values_xsqy = synthetic_values_xsqy[0];
    Kokkos::deep_copy(xsqy, values_xsqy);

    View2D j_long("j_long", numj, pver);

    View2D psum_l("psum_l", pver, nw);
    View2D psum_u("psum_u", pver, nw);

    auto team_policy = mam4::ThreadTeamPolicy(1u, mam4::testing::team_size);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          jlong(team, sza_in, alb_in, p_in, t_in, colo3_in, xsqy, sza, del_sza,
                alb, press, del_p, colo3, o3rat, del_alb, del_o3rat, etfphot,
                rsf_tab, // in
                prs, dprs, nw, nump, numsza, numcolo3, numalb, np_xs, numj,
                j_long, // output
                // work arrays
                rsf, psum_l, psum_u);
        });

    const Real zero = 0;
    std::vector<Real> jlong_out(pver * numj, zero);

    auto j_long_host = Kokkos::create_mirror_view(j_long);
    Kokkos::deep_copy(j_long_host, j_long);

    int count = 0;
    for (int j = 0; j < pver; ++j) {
      for (int i = 0; i < numj; ++i) {
        jlong_out[count] = j_long_host(i, j);
        count += 1;
      }
    }

    output.set("j_long", jlong_out);
  });
}
