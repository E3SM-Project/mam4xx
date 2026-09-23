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
void compute_ent_det_dp(Ensemble *ensemble) {
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
    Real dt = input.get("dt");
    EKAT_ASSERT(dt == 3600);

    std::vector<Real> dpdry_i_host, du_host, eu_host, ed_host, mu_i_host,
        md_i_host, eudp_host, dudp_host, eddp_host, dddp_host;
    mam4::ColumnView dpdry_i_dev, du_dev, eu_dev, ed_dev, mu_i_dev, md_i_dev,
        eudp_dev, dudp_dev, eddp_dev, dddp_dev, ntsub_dev;
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    get_input(input, "du", nlev, du_host, du_dev);
    get_input(input, "eu", nlev, eu_host, eu_dev);
    get_input(input, "ed", nlev, ed_host, ed_dev);
    get_input(input, "mu_i", nlev + 1, mu_i_host, mu_i_dev);
    get_input(input, "md_i", nlev + 1, md_i_host, md_i_dev);
    eudp_dev = mam4::validation::create_column_view(nlev);
    dudp_dev = mam4::validation::create_column_view(nlev);
    eddp_dev = mam4::validation::create_column_view(nlev);
    dddp_dev = mam4::validation::create_column_view(nlev);
    ntsub_dev = mam4::validation::create_column_view(1);
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Real dpdry_i[nlev];
          for (int i = 0; i < nlev; ++i)
            dpdry_i[i] = dpdry_i_dev[i];
          Real mu_i[nlev + 1];
          for (int i = 0; i < nlev + 1; ++i)
            mu_i[i] = mu_i_dev[i];
          Real md_i[nlev + 1];
          for (int i = 0; i < nlev + 1; ++i)
            md_i[i] = md_i_dev[i];
          Real du[nlev];
          for (int i = 0; i < nlev; ++i)
            du[i] = du_dev[i];
          Real eu[nlev];
          for (int i = 0; i < nlev; ++i)
            eu[i] = eu_dev[i];
          Real ed[nlev];
          for (int i = 0; i < nlev; ++i)
            ed[i] = ed_dev[i];
          int ntsub = 0;
          Real eudp[nlev], dudp[nlev], eddp[nlev], dddp[nlev];
          mam4::convproc::compute_ent_det_dp(team, nlev, ktop, kbot, dt, dpdry_i,
                                             mu_i, md_i, du, eu, ed, ntsub,
                                             eudp, dudp, eddp, dddp);
          for (int i = 0; i < nlev; ++i)
            eudp_dev[i] = eudp[i];
          for (int i = 0; i < nlev; ++i)
            dudp_dev[i] = dudp[i];
          for (int i = 0; i < nlev; ++i)
            eddp_dev[i] = eddp[i];
          for (int i = 0; i < nlev; ++i)
            dddp_dev[i] = dddp[i];
          ntsub_dev[0] = ntsub;
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_output(output, "eudp", nlev, eudp_host, eudp_dev);
    set_output(output, "dudp", nlev, dudp_host, dudp_dev);
    set_output(output, "eddp", nlev, eddp_host, eddp_dev);
    set_output(output, "dddp", nlev, dddp_host, dddp_dev);
    int ntsub = 0;
    {
      auto host_view = Kokkos::create_mirror_view(ntsub_dev);
      Kokkos::deep_copy(host_view, ntsub_dev);
      ntsub = host_view[0];
    }
    output.set("ntsub", ntsub);
  });
}
