// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void get_activate_frac(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    const Real zero = 0;
    const int ntot_amode = mam4::AeroConfig::num_modes();
    const int pcnst = mam4::aero_model::pcnst;

    const int pver = mam4::ndrop::pver;
    const auto state_q_db = input.get_array("state_q");

    const auto tair_db = input.get_array("temp");
    const auto pmid_db = input.get_array("pmid");
    const auto wsub_db = input.get_array("wsub");

    using View2D = mam4::ndrop::View2D;
    using View1DHost = typename mam4::HostType::view_1d<Real>;

    int count = 0;
    View2D state_q("state_q", pver, pcnst);
    auto state_host = Kokkos::create_mirror_view(state_q);

    for (int i = 0; i < pcnst; ++i) {
      // input data is store on the cpu.
      for (int kk = 0; kk < pver; ++kk) {
        state_host(kk, i) = state_q_db[count];
        count++;
      }
    }

    Kokkos::deep_copy(state_q, state_host);

    mam4::ColumnView tair;
    mam4::ColumnView pmid;
    mam4::ColumnView wsub;
    tair = mam4::testing::create_column_view(pver);
    pmid = mam4::testing::create_column_view(pver);
    wsub = mam4::testing::create_column_view(pver);

    auto tair_host = View1DHost((Real *)tair_db.data(), pver);
    auto pmid_host = View1DHost((Real *)pmid_db.data(), pver);
    auto wsub_host = View1DHost((Real *)wsub_db.data(), pver);

    Kokkos::deep_copy(tair, tair_host);
    Kokkos::deep_copy(pmid, pmid_host);
    Kokkos::deep_copy(wsub, wsub_host);

    View2D fn("fn", pver, ntot_amode);
    View2D fm("fm", pver, ntot_amode);
    View2D fluxn("fluxn", pver, ntot_amode);
    View2D fluxm("fluxm", pver, ntot_amode);
    mam4::ColumnView flux_fullact = mam4::testing::create_column_view(pver);

    Kokkos::parallel_for(
        "get_activate_frac", pver, KOKKOS_LAMBDA(int kk) {
          const Real air_density =
              mam4::conversions::density_of_ideal_gas(tair(kk), pmid(kk));
          // Note: Boltzmann’s constant and Avogadro’s number in
          // mam4::Constants have more digits than the e3sm values. Thus, aten
          // computed by ndrop_init has a relative difference of 1e-5 w.r.t
          // e3sm’s aten which causes this to test fail. I will use value of
          // aten from validation data only for testing proposes.
          Real aten_testing = 0.1206437615E-08;

          Real exp45logsig[mam4::AeroConfig::num_modes()],
              alogsig[mam4::AeroConfig::num_modes()],
              num2vol_ratio_min_nmodes[mam4::AeroConfig::num_modes()],
              num2vol_ratio_max_nmodes[mam4::AeroConfig::num_modes()] = {};

          Real aten2 = zero;

          mam4::ndrop::ndrop_init(
              exp45logsig, alogsig, aten2,
              num2vol_ratio_min_nmodes,  // voltonumbhi_amode
              num2vol_ratio_max_nmodes); // voltonumblo_amode

          const auto state_q_k = Kokkos::subview(state_q, kk, Kokkos::ALL());
          const auto fn_k = Kokkos::subview(fn, kk, Kokkos::ALL());
          const auto fm_k = Kokkos::subview(fm, kk, Kokkos::ALL());
          const auto fluxn_k = Kokkos::subview(fluxn, kk, Kokkos::ALL());
          const auto fluxm_k = Kokkos::subview(fluxm, kk, Kokkos::ALL());

          mam4::ndrop::get_activate_frac(
              state_q_k, air_density, air_density, wsub(kk),
              tair(kk), // in
              num2vol_ratio_min_nmodes, num2vol_ratio_max_nmodes, exp45logsig,
              alogsig, aten_testing, fn_k.data(), fm_k.data(), fluxn_k.data(),
              fluxm_k.data(), flux_fullact(kk));
        });

    auto fn_host = Kokkos::create_mirror_view(fn);
    Kokkos::deep_copy(fn_host, fn);

    auto fm_host = Kokkos::create_mirror_view(fm);
    Kokkos::deep_copy(fm_host, fm);

    auto fluxn_host = Kokkos::create_mirror_view(fluxn);
    Kokkos::deep_copy(fluxn_host, fluxn);

    auto fluxm_host = Kokkos::create_mirror_view(fluxm);
    Kokkos::deep_copy(fluxm_host, fluxm);

    std::vector<Real> host_v(pver);

    for (int i = 0; i < ntot_amode; ++i) {

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fn_host(kk, i);
      } // k

      output.set("fn_" + std::to_string(i + 1), host_v);

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fm_host(kk, i);
      } // k

      output.set("fm_" + std::to_string(i + 1), host_v);

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fluxn_host(kk, i);
      } // k

      output.set("fluxn_" + std::to_string(i + 1), host_v);

      for (int kk = 0; kk < pver; ++kk) {
        host_v[kk] = fluxm_host(kk, i);
      }
      output.set("fluxm_" + std::to_string(i + 1), host_v);
    } // i

    auto host = Kokkos::create_mirror_view(flux_fullact);

    Kokkos::deep_copy(host, flux_fullact);
    for (int kk = 0; kk < pver; ++kk) {
      host_v[kk] = host(kk);
    }

    output.set("flux_fullact", host_v);
  });
}
