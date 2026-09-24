// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <catch2/catch.hpp>

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
void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, std::vector<Real> &host,
               Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                            Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  EKAT_ASSERT(host.size() == rows * cols);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    // Col Major layout
    for (int i = 0, n = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j, ++n)
        matrix[i][j] = host[n];
    auto host_view = Kokkos::create_mirror_view(dev);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        host_view(i, j) = matrix[i][j];
    Kokkos::deep_copy(dev, host_view);
  }
}
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                                   Kokkos::MemoryUnmanaged> &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void compute_downdraft_mixing_ratio(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int pcnst_extd = mam4::ConvProc::pcnst_extd;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int ktop = input.get("ktop") - 1;
    EKAT_ASSERT(47 == ktop);
    const int kbot = input.get("kbot");
    EKAT_ASSERT(71 == kbot);

    std::vector<Real> doconvproc_extd_host, md_i_host, eddp_host, gath_host,
        cond_host;
    mam4::ColumnView doconvproc_extd_dev, md_i_dev, eddp_dev;
    Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged> gath_dev,
        cond_dev;

    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    get_input(input, "md_i", nlev + 1, md_i_host, md_i_dev);
    get_input(input, "eddp", nlev, eddp_host, eddp_dev);
    get_input(input, "const", nlev, pcnst_extd, gath_host, gath_dev);
    get_input(input, "cond", nlev + 1, pcnst_extd, cond_host, cond_dev);

    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool doconvproc_extd[pcnst_extd];
          Real md_i[nlev + 1];
          Real eddp[nlev];

          for (int i = 0; i < pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          for (int i = 0; i < nlev + 1; ++i)
            md_i[i] = md_i_dev(i);
          for (int i = 0; i < nlev; ++i)
            eddp[i] = eddp_dev(i);

          mam4::convproc::compute_downdraft_mixing_ratio(team, doconvproc_extd,
                                                         ktop, kbot, md_i, eddp,
                                                         gath_dev, cond_dev);
        });
    set_output(output, "cond", nlev + 1, pcnst_extd, cond_host, cond_dev);
  });
}
