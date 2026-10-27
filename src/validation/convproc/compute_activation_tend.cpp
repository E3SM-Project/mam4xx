// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <catch2/catch.hpp>

#include <mam4xx/convproc.hpp>
#include <validation.hpp>

using namespace skywalker;

namespace {
using View1D = typename mam4::DeviceType::view_1d<Real>;
using ViewInt1D = typename mam4::DeviceType::view_1d<int>;
using View1DHost = typename mam4::HostType::view_1d<Real>;
using ViewInt1DHost = typename mam4::HostType::view_1d<int>;

void get_input(const Input &input, const std::string &name, const int size,
               std::vector<Real> &host, View1D &dev) {
  host = input.get_array(name);
  dev = View1D(name, size);

  EKAT_ASSERT(host.size() == size);
  auto host_view = View1DHost((Real *)host.data(), size);
  Kokkos::deep_copy(dev, host_view);
}

void set_output(Output &output, const std::string &name, const int size,
                std::vector<Real> &host, const View1D &dev) {
  host.resize(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0; i < size; ++i) {
    host[i] = host_view[i];
  }
  output.set(name, host);
}
} // namespace

void compute_activation_tend(Ensemble *ensemble) {
  auto aero_species =
      mam4::aero_species_on_device(mam4::default_aero_species());

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int pcnst_extd = mam4::ConvProc::pcnst_extd;

    // Fetch scalar parameters from array format
    std::vector<Real> kk_arr = input.get_array("kk");
    const int kk = (int)kk_arr[0] - 1;

    std::vector<Real> kactcnt_arr = input.get_array("kactcnt");
    int kactcnt = (int)kactcnt_arr[0];

    std::vector<Real> kactfirst_arr = input.get_array("kactfirst");
    int kactfirst = (int)kactfirst_arr[0] - 1;

    std::vector<Real> f_ent_arr = input.get_array("f_ent");
    const Real f_ent = f_ent_arr[0];

    std::vector<Real> xx_wcldbase_arr = input.get_array("xx_wcldbase_icol");
    Real xx_wcldbase = xx_wcldbase_arr[0];

    std::vector<Real> xx_kcldbase_arr = input.get_array("xx_kcldbase_icol");
    int xx_kcldbase = (int)xx_kcldbase_arr[0];

    // Fetch single-level input scalars
    std::vector<Real> cldfrac_i_kk = input.get_array("cldfrac_i_kk");
    std::vector<Real> rhoair_i_kk = input.get_array("rhoair_i_kk");
    std::vector<Real> mu_i_kk = input.get_array("mu_i_kk");
    std::vector<Real> mu_i_kp1 = input.get_array("mu_i_kp1");
    std::vector<Real> dt_u_kk = input.get_array("dt_u_kk");
    std::vector<Real> wup_kk = input.get_array("wup_kk");
    std::vector<Real> temperature_kk = input.get_array("temperature_kk");
    std::vector<Real> icwmr_kk = input.get_array("icwmr_kk");

    // Fetch array inputs and move to device
    std::vector<Real> conu_host, dconudt_activa_host;
    View1D conu_dev, dconudt_activa_dev;
    get_input(input, "conu_kk", pcnst_extd, conu_host, conu_dev);
    get_input(input, "dconudt_activa_kk", pcnst_extd, dconudt_activa_host,
              dconudt_activa_dev);

    // Create device variables for output scalars
    ViewInt1D kactcnt_dev("kactcnt", 1);
    ViewInt1D kactfirst_dev("kactfirst", 1);
    View1D xx_wcldbase_dev("xx_wcldbase", 1);
    ViewInt1D xx_kcldbase_dev("xx_kcldbase", 1);

    auto kactcnt_host_view = Kokkos::create_mirror_view(kactcnt_dev);
    auto kactfirst_host_view = Kokkos::create_mirror_view(kactfirst_dev);
    auto xx_wcldbase_host_view = Kokkos::create_mirror_view(xx_wcldbase_dev);
    auto xx_kcldbase_host_view = Kokkos::create_mirror_view(xx_kcldbase_dev);

    kactcnt_host_view(0) = kactcnt;
    kactfirst_host_view(0) = kactfirst;
    xx_wcldbase_host_view(0) = xx_wcldbase;
    xx_kcldbase_host_view(0) = xx_kcldbase;

    Kokkos::deep_copy(kactcnt_dev, kactcnt_host_view);
    Kokkos::deep_copy(kactfirst_dev, kactfirst_host_view);
    Kokkos::deep_copy(xx_wcldbase_dev, xx_wcldbase_host_view);
    Kokkos::deep_copy(xx_kcldbase_dev, xx_kcldbase_host_view);

    // Call function within parallel_for for GPU compatibility
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          // Extract scalars from device views to local variables
          int kactcnt_local = kactcnt_dev(0);
          int kactfirst_local = kactfirst_dev(0);
          Real xx_wcldbase_local = xx_wcldbase_dev(0);
          int xx_kcldbase_local = xx_kcldbase_dev(0);

          mam4::convproc::compute_activation_tend(
              aero_species, f_ent, cldfrac_i_kk[0], rhoair_i_kk[0], mu_i_kk[0],
              mu_i_kp1[0], dt_u_kk[0], wup_kk[0], icwmr_kk[0],
              temperature_kk[0], kk, kactcnt_local, kactfirst_local, conu_dev,
              dconudt_activa_dev, xx_wcldbase_local, xx_kcldbase_local);

          // Copy modified scalars back to device views
          kactcnt_dev(0) = kactcnt_local;
          kactfirst_dev(0) = kactfirst_local;
          xx_wcldbase_dev(0) = xx_wcldbase_local;
          xx_kcldbase_dev(0) = xx_kcldbase_local;
        });
    Kokkos::fence();

    // Copy results back to host
    Kokkos::deep_copy(kactcnt_host_view, kactcnt_dev);
    Kokkos::deep_copy(kactfirst_host_view, kactfirst_dev);
    Kokkos::deep_copy(xx_wcldbase_host_view, xx_wcldbase_dev);
    Kokkos::deep_copy(xx_kcldbase_host_view, xx_kcldbase_dev);

    set_output(output, "conu_kk", pcnst_extd, conu_host, conu_dev);
    set_output(output, "dconudt_activa_kk", pcnst_extd, dconudt_activa_host,
               dconudt_activa_dev);
    output.set("kactcnt", (int)kactcnt_host_view(0));
    output.set("kactfirst", 1 + (int)kactfirst_host_view(0));
    output.set("xx_wcldbase_icol", xx_wcldbase_host_view(0));
    output.set("xx_kcldbase_icol", (int)xx_kcldbase_host_view(0));
  });
}
