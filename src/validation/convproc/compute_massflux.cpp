// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <ekat_assert.hpp>

#include <mam4xx/convproc.hpp>
#include <validation.hpp>

using namespace skywalker;

namespace {
void get_input(const Input &input, const std::string &name, const int size,
               std::vector<Real> &host, mam4::ColumnView &dev) {
  host = input.get_array(name);
  EKAT_ASSERT(host.size() == size);
  dev = mam4::validation::create_column_view(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  for (int n = 0; n < size; ++n)
    host_view[n] = host[n];
  Kokkos::deep_copy(dev, host_view);
}
void set_output(Output &output, const std::string &name, const int size,
                std::vector<Real> &host, const mam4::ColumnView &dev) {
  host.resize(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int n = 0; n < size; ++n)
    host[n] = host_view[n];
  output.set(name, host);
}
} // namespace
void compute_massflux(Ensemble *ensemble) {
  using View1D = Kokkos::View<Real *>;
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int ktop = input.get("ktop") - 1;
    EKAT_ASSERT(ktop == 47);
    const int kbot = input.get("kbot");
    EKAT_ASSERT(kbot == 71);
    Real xx_mfup_max = input.get("xx_mfup_max");
    EKAT_ASSERT(xx_mfup_max == 0);

    std::vector<Real> dpdry_i_host, du_host, eu_host, ed_host, mu_i_host,
        md_i_host;
    mam4::ColumnView dpdry_i_dev, du_dev, eu_dev, ed_dev, xx_mfup_max_dev;
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    get_input(input, "du", nlev, du_host, du_dev);
    get_input(input, "eu", nlev, eu_host, eu_dev);
    get_input(input, "ed", nlev, ed_host, ed_dev);
    View1D mu_i_dev("mu_i_dev", nlev + 1);
    View1D md_i_dev("md_i_dev", nlev + 1);
    xx_mfup_max_dev = mam4::validation::create_column_view(1);
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          mam4::convproc::compute_massflux(team, nlev, ktop, kbot, dpdry_i_dev,
                                           du_dev, eu_dev, ed_dev, mu_i_dev,
                                           md_i_dev, xx_mfup_max_dev[0]);
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_output(output, "mu_i", nlev + 1, mu_i_host, mu_i_dev);
    set_output(output, "md_i", nlev + 1, md_i_host, md_i_dev);
    {
      auto host_view = Kokkos::create_mirror_view(xx_mfup_max_dev);
      Kokkos::deep_copy(host_view, xx_mfup_max_dev);
      xx_mfup_max = host_view[0];
    }
    output.set("xx_mfup_max", xx_mfup_max);
  });
}
