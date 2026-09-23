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
void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, std::vector<Real> &host,
               Kokkos::View<Real * [mam4::aero_model::pcnst],
                            Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  EKAT_ASSERT(host.size() == rows * cols);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    for (int j = 0, n = 0; j < cols; ++j)
      for (int i = 0; i < rows; ++i, ++n)
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
void initialize_tmr_array(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int pcnst_extd = mam4::ConvProc::pcnst_extd;
    const int pcnst = mam4::aero_model::pcnst;
    using Kokko2DView =
        Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>;
    // Fetch ensemble parameters
    EKAT_ASSERT(pcnst_extd == input.get("pcnst_extd"));
    EKAT_ASSERT(pcnst == input.get("ncnst"));
    const int iconvtype = input.get("iconvtype");

    std::vector<Real> doconvproc_extd_host, q_i_host, gath_host, chat_host,
        conu_host, cond_host;
    mam4::ColumnView doconvproc_extd_dev;
    Kokkos::View<Real *[pcnst], Kokkos::MemoryUnmanaged> q_i_dev;
    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    get_input(input, "q_i", nlev, pcnst, q_i_host, q_i_dev);

    auto col_view = mam4::validation::create_column_view(pcnst_extd * nlev);
    Kokko2DView gath_dev(col_view.data(), nlev, pcnst_extd);
    col_view = mam4::validation::create_column_view(pcnst_extd * (nlev + 1));
    Kokko2DView chat_dev(col_view.data(), nlev + 1, pcnst_extd);
    col_view = mam4::validation::create_column_view(pcnst_extd * (nlev + 1));
    Kokko2DView conu_dev(col_view.data(), nlev + 1, pcnst_extd);
    col_view = mam4::validation::create_column_view(pcnst_extd * (nlev + 1));
    Kokko2DView cond_dev(col_view.data(), nlev + 1, pcnst_extd);
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool doconvproc_extd[pcnst_extd];
          for (int i = 0; i < pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          mam4::convproc::initialize_tmr_array(team, nlev, iconvtype, doconvproc_extd,
                                               q_i_dev, gath_dev, chat_dev,
                                               conu_dev, cond_dev);
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_output(output, "const", nlev, pcnst_extd, gath_host, gath_dev);
    set_output(output, "chat", nlev + 1, pcnst_extd, chat_host, chat_dev);
    set_output(output, "conu", nlev + 1, pcnst_extd, conu_host, conu_dev);
    set_output(output, "cond", nlev + 1, pcnst_extd, cond_host, cond_dev);
  });
}
