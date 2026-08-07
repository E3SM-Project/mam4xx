// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::ndrop;

void update_from_cldn_profile(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = mam4::HostType::view_1d<Real>;
    using View1D = mam4::ndrop::View1D;
    const Real zero = 0;
    const int ntot_amode = mam4::AeroConfig::num_modes();
    const int ncnst_tot = mam4::ndrop::ncnst_tot;

    const Real cldn_col_in = input.get_array("cldn_col_in_kk")[0];
    const Real cldn_col_in_kp1 = input.get_array("cldn_col_in_kp1")[0];
    const Real dtinv = input.get_array("dtinv")[0];
    const Real wtke_col_in = input.get_array("wtke_col_in")[0];
    const Real temp_col_in = input.get_array("temp_col_in")[0];
    const Real air_density = input.get_array("cs_col_in_kk")[0];
    const Real air_density_kp1 = input.get_array("cs_col_in_kp1")[0];

    auto state_q_col_in_kp1 = input.get_array("state_q_col_in_kp1");
    Real qcld = input.get_array("qcld")[0];

    Real exp45logsig[mam4::AeroConfig::num_modes()],
        alogsig[mam4::AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[mam4::AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[mam4::AeroConfig::num_modes()] = {};

    Real aten = zero;

    mam4::ndrop::ndrop_init(exp45logsig, alogsig, aten,
                            num2vol_ratio_min_nmodes,  // voltonumbhi_amode
                            num2vol_ratio_max_nmodes); // voltonumblo_amode

    auto raercol_nsav = input.get_array("raercol_nsav_kk");
    auto raercol_nsav_kp1 = input.get_array("raercol_nsav_kp1");
    auto raercol_cw_nsav = input.get_array("raercol_cw_nsav");
    auto nsource_col = input.get_array("nsource_col")[0];
    auto factnum_col = input.get_array("factnum_col");
    auto nact = input.get_array("nact");
    auto mact = input.get_array("mact");

    Real ekd = input.get_array("ekd")[0];
    const Real csbot_cscen = input.get_array("csbot_cscen")[0];
    auto dz = input.get_array("dz")[0];
    auto zs = input.get_array("zs")[0];

    View1DHost raercol_nsav_host(raercol_nsav.data(), ncnst_tot);
    View1DHost raercol_nsav_kp1_host(raercol_nsav_kp1.data(), ncnst_tot);
    View1DHost raercol_cw_nsav_host(raercol_cw_nsav.data(), ncnst_tot);
    View1D raercol_nsav_view("raercol_nsav_view", ncnst_tot);
    View1D raercol_nsav_kp1_view("raercol_nsav_kp1_view", ncnst_tot);
    View1D raercol_cw_nsav_view("raercol_cw_nsav_view", ncnst_tot);
    Kokkos::deep_copy(raercol_nsav_view, raercol_nsav_host);
    Kokkos::deep_copy(raercol_nsav_kp1_view, raercol_nsav_kp1_host);
    Kokkos::deep_copy(raercol_cw_nsav_view, raercol_cw_nsav_host);

    View1DHost nsource_col_host(&nsource_col, 1);
    View1D nsource_col_view("nsource_col_view", 1);
    Kokkos::deep_copy(nsource_col_view, nsource_col_host);

    View1DHost qcld_host(&qcld, 1);
    View1D qcld_view("qcld_view", 1);
    Kokkos::deep_copy(qcld_view, qcld_host);

    View1DHost factnum_col_host(factnum_col.data(), ntot_amode);
    View1D factnum_col_view("factnum_col_view", ntot_amode);
    Kokkos::deep_copy(factnum_col_view, factnum_col_host);

    View1DHost ekd_host(&ekd, 1);
    View1D ekd_view("ekd_view", 1);
    Kokkos::deep_copy(ekd_view, ekd_host);

    View1DHost nact_host(nact.data(), ntot_amode);
    View1D nact_view("nact_view", ntot_amode);
    Kokkos::deep_copy(nact_view, nact_host);

    View1DHost mact_host(mact.data(), ntot_amode);
    View1D mact_view("mact_view", ntot_amode);
    Kokkos::deep_copy(mact_view, mact_host);

    const int pcnst = mam4::aero_model::pcnst;
    View1DHost state_q_col_in_kp1_host(state_q_col_in_kp1.data(), pcnst);
    View1D state_q_col_in_kp1_view("state_q_col_in_kp1_view", pcnst);
    Kokkos::deep_copy(state_q_col_in_kp1_view, state_q_col_in_kp1_host);

    Kokkos::parallel_for(
        "update_from_cldn_profile", 1, KOKKOS_LAMBDA(int) {
          mam4::ndrop::update_from_cldn_profile(
              cldn_col_in, cldn_col_in_kp1, dtinv, wtke_col_in, zs,
              dz, // ! in
              temp_col_in, air_density, air_density_kp1, csbot_cscen,
              state_q_col_in_kp1_view.data(), // ! in
              num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes, exp45logsig,
              alogsig, aten, raercol_nsav_view, raercol_nsav_kp1_view,
              raercol_cw_nsav_view,
              nsource_col_view[0], // inout
              qcld_view[0], factnum_col_view.data(),
              ekd_view[0], // out
              nact_view, mact_view);
        });

    Kokkos::deep_copy(qcld_host, qcld_view);
    qcld = qcld_host[0];

    Kokkos::deep_copy(nsource_col_host, nsource_col_view);
    nsource_col = nsource_col_host[0];

    Kokkos::deep_copy(raercol_nsav_host, raercol_nsav_view);
    Kokkos::deep_copy(raercol_cw_nsav_host, raercol_cw_nsav_view);
    Kokkos::deep_copy(factnum_col_host, factnum_col_view);

    for (int i = 0; i < ncnst_tot; ++i)
      raercol_nsav[i] = raercol_nsav_host[i];
    for (int i = 0; i < ncnst_tot; ++i)
      raercol_cw_nsav[i] = raercol_cw_nsav_host[i];
    for (int i = 0; i < ntot_amode; ++i)
      factnum_col[i] = factnum_col_host[i];

    Kokkos::deep_copy(ekd_host, ekd_view);
    ekd = ekd_host[0];

    Kokkos::deep_copy(nact_host, nact_view);
    Kokkos::deep_copy(mact_host, mact_view);
    for (int i = 0; i < ntot_amode; ++i)
      nact[i] = nact_host[i];
    for (int i = 0; i < ntot_amode; ++i)
      mact[i] = mact_host[i];

    output.set("qcld", qcld);
    output.set("nsource_col", nsource_col);
    output.set("raercol_nsav", raercol_nsav);
    output.set("raercol_cw_nsav", raercol_cw_nsav);
    output.set("factnum_col", factnum_col);
    output.set("ekd", ekd);
    output.set("nact", nact);
    output.set("mact", mact);
  });
}
