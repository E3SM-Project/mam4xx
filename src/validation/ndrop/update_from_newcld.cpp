// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::ndrop;

void update_from_newcld(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = mam4::HostType::view_1d<Real>;
    using View1D = mam4::ndrop::View1D;
    // number of vertical points
    // validation test from standalone ndrop
    const Real zero = 0;
    const int ntot_amode = mam4::AeroConfig::num_modes();
    const int ncnst_tot = mam4::ndrop::ncnst_tot;

    const Real cldn_col_in = input.get_array("cldn_col_in")[0];
    const Real cldo_col_in = input.get_array("cldo_col_in")[0];
    const Real dtinv = input.get_array("dtinv")[0];
    const Real wtke_col_in = input.get_array("wtke_col_in")[0];
    const Real temp_col_in = input.get_array("temp_col_in")[0];
    const Real air_density = input.get_array("cs_col_in")[0];
    auto state_q = input.get_array("state_q_col_in");
    Real qcld = input.get_array("qcld")[0];

    Real exp45logsig[mam4::AeroConfig::num_modes()],
        alogsig[mam4::AeroConfig::num_modes()],
        num2vol_ratio_min_nmodes[mam4::AeroConfig::num_modes()],
        num2vol_ratio_max_nmodes[mam4::AeroConfig::num_modes()] = {};

    Real aten = zero;

    ndrop_init(exp45logsig, alogsig, aten,
               num2vol_ratio_min_nmodes,  // voltonumbhi_amode
               num2vol_ratio_max_nmodes); // voltonumblo_amode

    auto raercol_nsav = input.get_array("raercol_nsav");
    auto raercol_cw_nsav = input.get_array("raercol_cw_nsav");
    auto nsource_col_out = input.get_array("nsource_col_out")[0];
    auto factnum_col_out = input.get_array("factnum_col_out");

    View1DHost raercol_nsav_host(raercol_nsav.data(), ncnst_tot);
    View1DHost raercol_cw_nsav_host(raercol_cw_nsav.data(), ncnst_tot);
    View1D raercol_nsav_view("raercol_nsav_view", ncnst_tot);
    View1D raercol_cw_nsav_view("raercol_cw_nsav_view", ncnst_tot);
    Kokkos::deep_copy(raercol_nsav_view, raercol_nsav_host);
    Kokkos::deep_copy(raercol_cw_nsav_view, raercol_cw_nsav_host);

    View1DHost qcld_host(&qcld, 1);
    View1D qcld_view("qcld_view", 1);
    Kokkos::deep_copy(qcld_view, qcld_host);

    View1DHost nsource_col_out_host(&nsource_col_out, 1);
    View1D nsource_col_out_view("nsource_col_out_view", 1);
    Kokkos::deep_copy(nsource_col_out_view, nsource_col_out_host);

    View1DHost factnum_col_out_host(factnum_col_out.data(), ntot_amode);
    View1D factnum_col_out_view("factnum_col_out_view", ntot_amode);
    Kokkos::deep_copy(factnum_col_out_view, factnum_col_out_host);

    const int pcnst = mam4::aero_model::pcnst;
    View1DHost state_q_host(state_q.data(), pcnst);
    View1D state_q_view("state_q_view", pcnst);
    Kokkos::deep_copy(state_q_view, state_q_host);

    Kokkos::parallel_for(
        "update_from_newcld", 1, KOKKOS_LAMBDA(int) {
          update_from_newcld(
              cldn_col_in, cldo_col_in, dtinv, //& ! in
              wtke_col_in, temp_col_in, air_density,
              state_q_view, //& ! in
              num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes, exp45logsig,
              alogsig, aten, qcld_view[0], raercol_nsav_view,
              raercol_cw_nsav_view, //&      ! inout
              nsource_col_out_view[0], factnum_col_out_view.data());
        });
    Kokkos::deep_copy(raercol_nsav_host, raercol_nsav_view);
    Kokkos::deep_copy(raercol_cw_nsav_host, raercol_cw_nsav_view);
    Kokkos::deep_copy(qcld_host, qcld_view);
    Kokkos::deep_copy(nsource_col_out_host, nsource_col_out_view);
    Kokkos::deep_copy(factnum_col_out_host, factnum_col_out_view);
    for (int i = 0; i < ncnst_tot; ++i)
      raercol_nsav[i] = raercol_nsav_host[i];
    for (int i = 0; i < ncnst_tot; ++i)
      raercol_cw_nsav[i] = raercol_cw_nsav_host[i];
    for (int i = 0; i < ntot_amode; ++i)
      factnum_col_out[i] = factnum_col_out_host[i];
    qcld = qcld_host[0];
    nsource_col_out = nsource_col_out_host[0];
    output.set("qcld", qcld);
    output.set("nsource_col_out", nsource_col_out);
    output.set("raercol_nsav", raercol_nsav);
    output.set("raercol_cw_nsav", raercol_cw_nsav);
    output.set("factnum_col_out", factnum_col_out);
  });
}
