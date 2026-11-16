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
void compute_midlev_height(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;

    std::vector<Real> dpdry_i_host, rhoair_i_host, zmagl_host;
    mam4::ColumnView dpdry_i_dev, rhoair_i_dev, zmagl_dev;
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    get_input(input, "rhoair_i", nlev, rhoair_i_host, rhoair_i_dev);
    zmagl_dev = mam4::validation::create_column_view(nlev);
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Real dpdry_i[nlev];
          for (int i = 0; i < nlev; ++i)
            dpdry_i[i] = dpdry_i_dev[i];
          Real rhoair_i[nlev];
          for (int i = 0; i < nlev; ++i)
            rhoair_i[i] = rhoair_i_dev[i];
          Real zmagl[nlev];
          mam4::convproc::compute_midlev_height(team, nlev, dpdry_i, rhoair_i,
                                                zmagl);
          for (int i = 0; i < nlev; ++i)
            zmagl_dev[i] = zmagl[i];
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_output(output, "zmagl", nlev, zmagl_host, zmagl_dev);
  });
}
