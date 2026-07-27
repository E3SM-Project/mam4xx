// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void dropmixnuc(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const int ntot_amode = mam4::AeroConfig::num_modes();
    const int pcnst = mam4::aero_model::pcnst;
    const int psat = mam4::ndrop::psat;
    const int ncnst_tot = mam4::ndrop::ncnst_tot;

    const int pver = mam4::ndrop::pver;

    const auto state_q_db = input.get_array("state_q");
    // Conversion from Fortran to C++ indexing.
    const int top_lev = static_cast<int>(input.get_array("top_lev")[0]) - 1;

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");
    const auto pint_db = input.get_array("pint");

    const auto pdel_db = input.get_array("pdel");
    const auto rpdel_db = input.get_array("rpdel");
    const auto zm_db = input.get_array("zm");
    const auto ncldwtr_db = input.get_array("ncldwtr");
    const auto kvh_db = input.get_array("kvh");
    const auto cldn_db = input.get_array("cldn");
    const auto wsub_db = input.get_array("wsub");
    const auto cldo_db = input.get_array("cldo");
    const auto qqcw_db = input.get_array("qqcw");

    using View1D = mam4::ndrop::View1D;
    using View2D = mam4::ndrop::View2D;
    using View3D = mam4::ndrop::View3D;

    using View1DHost = typename mam4::HostType::view_1d<Real>;

    View2D state_q("state_q", pver, pcnst);
    auto state_host = Kokkos::create_mirror_view(state_q);

    for (int i = 0, count = 0; i < pcnst; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk, ++count) {
        state_host(kk, i) = state_q_db[count];
      }
    }

    Kokkos::deep_copy(state_q, state_host);

    View2D qqcw("qqcw", ncnst_tot, pver);
    auto qqcw_host = Kokkos::create_mirror_view(qqcw);

    for (int kk = 0, count = 0; kk < pver; ++kk) {
      for (int i = 0; i < ncnst_tot; ++i, ++count) {
        qqcw_host(i, kk) = qqcw_db[count];
      }
    }

    // transfer data to GPU.
    Kokkos::deep_copy(qqcw, qqcw_host);

    mam4::ColumnView tair;
    mam4::ColumnView pmid;
    mam4::ColumnView pint;
    mam4::ColumnView pdel;
    mam4::ColumnView rpdel;
    mam4::ColumnView zm;
    mam4::ColumnView ncldwtr;
    mam4::ColumnView kvh;
    mam4::ColumnView cldn;
    mam4::ColumnView wsub;
    mam4::ColumnView cldo;
    tair = mam4::testing::create_column_view(pver);
    pmid = mam4::testing::create_column_view(pver);
    pint = mam4::testing::create_column_view(pver);
    pdel = mam4::testing::create_column_view(pver);
    rpdel = mam4::testing::create_column_view(pver);
    zm = mam4::testing::create_column_view(pver);
    ncldwtr = mam4::testing::create_column_view(pver);
    kvh = mam4::testing::create_column_view(pver);
    cldn = mam4::testing::create_column_view(pver);
    wsub = mam4::testing::create_column_view(pver);
    cldo = mam4::testing::create_column_view(pver);

    auto tair_host = View1DHost((Real *)tair_db.data(), pver);
    auto pmid_host = View1DHost((Real *)pmid_db.data(), pver);
    auto pint_host = View1DHost((Real *)pint_db.data(), pver);
    auto pdel_host = View1DHost((Real *)pdel_db.data(), pver);
    auto rpdel_host = View1DHost((Real *)rpdel_db.data(), pver);
    auto zm_host = View1DHost((Real *)zm_db.data(), pver);
    auto ncldwtr_host = View1DHost((Real *)ncldwtr_db.data(), pver);
    auto kvh_host = View1DHost((Real *)kvh_db.data(), pver);
    auto cldn_host = View1DHost((Real *)cldn_db.data(), pver);
    auto wsub_host = View1DHost((Real *)wsub_db.data(), pver);
    auto cldo_host = View1DHost((Real *)cldo_db.data(), pver);

    Kokkos::deep_copy(tair, tair_host);
    Kokkos::deep_copy(pmid, pmid_host);
    Kokkos::deep_copy(pint, pint_host);

    Kokkos::deep_copy(pdel, pdel_host);
    Kokkos::deep_copy(rpdel, rpdel_host);
    Kokkos::deep_copy(zm, zm_host);
    Kokkos::deep_copy(ncldwtr, ncldwtr_host);
    Kokkos::deep_copy(kvh, kvh_host);
    Kokkos::deep_copy(cldn, cldn_host);
    Kokkos::deep_copy(wsub, wsub_host);
    Kokkos::deep_copy(cldo, cldo_host);

    const Real dtmicro = input.get_array("dtmicro")[0];

    // output
    mam4::ColumnView qcld;
    mam4::ColumnView tendnd;
    mam4::ColumnView ndropcol;
    mam4::ColumnView ndropmix;
    mam4::ColumnView nsource;
    mam4::ColumnView wtke;
    qcld = mam4::testing::create_column_view(pver);
    tendnd = mam4::testing::create_column_view(pver);
    ndropcol = mam4::testing::create_column_view(pver);
    ndropmix = mam4::testing::create_column_view(pver);
    nsource = mam4::testing::create_column_view(pver);
    wtke = mam4::testing::create_column_view(pver);

    View2D ptend_q("ptend_q", pcnst, pver);
    View2D factnum("factnum", ntot_amode, pver);
    View2D coltend("coltend", ncnst_tot, pver);
    View2D coltend_cw("coltend_cw", ncnst_tot, pver);
    View2D ccn("ccn", pver, psat);
    View3D raercol_cw("raercol_cw", 2, ncnst_tot, pver);
    View3D raercol("raercol", 2, ncnst_tot, pver);

    View2D nact("nact", pver, ntot_amode);
    View2D mact("mact", pver, ntot_amode);

    mam4::ColumnView ekd;
    ekd = mam4::testing::create_column_view(pver);

    mam4::ColumnView zn, csbot, zs, overlapp, overlapm, ekkp, ekkm, qncld, srcn,
        source;

    zn = mam4::testing::create_column_view(pver);
    csbot = mam4::testing::create_column_view(pver);
    zs = mam4::testing::create_column_view(pver);
    overlapp = mam4::testing::create_column_view(pver);
    overlapm = mam4::testing::create_column_view(pver);
    ekkp = mam4::testing::create_column_view(pver);
    ekkm = mam4::testing::create_column_view(pver);
    qncld = mam4::testing::create_column_view(pver);
    srcn = mam4::testing::create_column_view(pver);
    source = mam4::testing::create_column_view(pver);

    mam4::ColumnView dz, csbot_cscen;
    dz = mam4::testing::create_column_view(pver);
    csbot_cscen = mam4::testing::create_column_view(pver);

    mam4::ColumnView raertend, qqcwtend;

    raertend = mam4::testing::create_column_view(pver);
    qqcwtend = mam4::testing::create_column_view(pver);

    auto team_policy = mam4::ThreadTeamPolicy(1u, mam4::testing::team_size);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Real exp45logsig[mam4::AeroConfig::num_modes()],
              alogsig[mam4::AeroConfig::num_modes()],
              num2vol_ratio_min_nmodes[mam4::AeroConfig::num_modes()],
              num2vol_ratio_max_nmodes[mam4::AeroConfig::num_modes()] = {};

          Real aten = zero;

          mam4::ndrop::ndrop_init(
              exp45logsig, alogsig, aten,
              num2vol_ratio_min_nmodes,  // voltonumbhi_amode
              num2vol_ratio_max_nmodes); // voltonumblo_amode
          mam4::ndrop::dropmixnuc(
              team, dtmicro, tair, pmid, pint, pdel, rpdel,
              zm, //  ! in zm[kk] - zm[kk+1], for pver zm[kk-1] - zm[kk]
              state_q, ncldwtr,
              kvh, // kvh[kk+1]
              cldn, num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes,
              exp45logsig, alogsig, aten, true,
              qcld, //
              wsub,
              cldo, // in
              qqcw, // inout
              ptend_q, tendnd, factnum, ndropcol, ndropmix, nsource, wtke, ccn,
              coltend, coltend_cw, top_lev, raercol_cw, raercol, nact, mact,
              ekd,
              // work arrays
              zn, csbot, zs, overlapp, overlapm, ekkp, ekkm, qncld, srcn,
              source, dz, csbot_cscen, raertend, qqcwtend);
        });

    auto host = Kokkos::create_mirror_view(tendnd);
    Kokkos::deep_copy(host, tendnd);
    std::vector<Real> v_host(pver);
    for (int kk = 0; kk < pver; ++kk) {
      v_host[kk] = host(kk);
    }

    output.set("tendnd", v_host);

    std::vector<Real> output_qqcw;

    // transfer data to host
    Kokkos::deep_copy(qqcw_host, qqcw);

    for (int kk = 0; kk < pver; ++kk) {
      for (int i = 0; i < ncnst_tot; ++i) {
        output_qqcw.push_back(qqcw_host(i, kk));
      }
    }

    output.set("qqcw", output_qqcw);

    auto ptend_q_host = Kokkos::create_mirror_view(ptend_q);
    Kokkos::deep_copy(ptend_q_host, ptend_q);
    std::vector<Real> output_ptend_q;
    for (int i = 0; i < pcnst; ++i) {
      for (int kk = 0; kk < pver; ++kk) {
        output_ptend_q.push_back(ptend_q_host(i, kk));
      }
    }
    output.set("ptend_q", output_ptend_q);

    auto factnum_host = Kokkos::create_mirror_view(factnum);
    Kokkos::deep_copy(factnum_host, factnum);

    std::vector<Real> output_factnum;
    for (int i = 0; i < ntot_amode; ++i) {
      for (int kk = 0; kk < pver; ++kk) {
        output_factnum.push_back(factnum_host(i, kk));
      }
    }

    output.set("factnum", output_factnum);
  });
}
