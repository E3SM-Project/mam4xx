// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <ekat_assert.hpp>

#include <mam4xx/convproc.hpp>
#include <validation.hpp>

using namespace skywalker;

namespace {
using View1D = typename mam4::DeviceType::view_1d<Real>;
using View1DHost = typename mam4::HostType::view_1d<Real>;

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

void ma_activate_convproc(Ensemble *ensemble) {
  auto aero_species =
      mam4::aero_species_on_device(mam4::default_aero_species());

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int pcnst_extd = mam4::ConvProc::pcnst_extd;

    // Fetch scalar parameters from array format
    std::vector<Real> f_ent_arr = input.get_array("f_ent");
    const Real f_ent = f_ent_arr[0];
    std::vector<Real> dt_u_arr = input.get_array("dt_u");
    const Real dt_u = dt_u_arr[0];
    std::vector<Real> wup_arr = input.get_array("wup");
    const Real wup = wup_arr[0];
    std::vector<Real> tair_arr = input.get_array("tair");
    const Real tair = tair_arr[0];
    std::vector<Real> rhoair_arr = input.get_array("rhoair");
    const Real rhoair = rhoair_arr[0];
    std::vector<Real> kk_arr = input.get_array("kk");
    const int kk = (int)kk_arr[0] - 1;
    std::vector<Real> kactfirst_arr = input.get_array("kactfirst");
    const int kactfirst = (int)kactfirst_arr[0] - 1;

    // Fetch array inputs and move to device
    std::vector<Real> conu_host, dconudt_host;
    View1D conu_dev, dconudt_dev;
    get_input(input, "conu", pcnst_extd, conu_host, conu_dev);
    get_input(input, "dconudt", pcnst_extd, dconudt_host, dconudt_dev);

    // Call function within parallel_for for GPU compatibility
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          mam4::convproc::ma_activate_convproc(
              aero_species, conu_dev.data(), dconudt_dev.data(), f_ent, dt_u,
              wup, tair, rhoair, kk, kactfirst);
        });
    Kokkos::fence();

    // Copy results back to host
    set_output(output, "conu", pcnst_extd, conu_host, conu_dev);
    set_output(output, "dconudt", pcnst_extd, dconudt_host, dconudt_dev);
  });
}
