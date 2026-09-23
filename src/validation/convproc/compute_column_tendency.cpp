// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/convproc.hpp>
#include <validation.hpp>

using namespace skywalker;

namespace {
void get_input(const Input &input, const std::string &name, const int size,
               std::vector<Real> &host, mam4::ColumnView &dev) {
  host = input.get_array(name);
  dev = mam4::validation::create_column_view(size);
  EKAT_ASSERT(host.size() == size);
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
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                     Kokkos::MemoryUnmanaged>(col_view.data(), rows, cols);
  EKAT_ASSERT(host.size() == rows * cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    // Row Major layout
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
void set_output(Output &output, const std::string &name, const int size,
                std::vector<Real> &host, const mam4::ColumnView &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int n = 0; n < size; ++n)
    host[n] = host_view[n];
  output.set(name, host);
}
} // namespace
void compute_column_tendency(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int nlevp = 1 + nlev;
    const int pcnst_extd = mam4::ConvProc::pcnst_extd;
    // Fetch ensemble parameters

    // these variables depend on mode No and k
    const int ktop = input.get("ktop") - 1; // Make C++ offset
    const int kbot_prevap = input.get("kbot_prevap");
    EKAT_ASSERT(nlev == kbot_prevap);

    // flag for doing convective transport
    std::vector<Real> doconvproc_extd_host;
    mam4::ColumnView doconvproc_extd_dev;
    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    std::vector<Real> dconudt_activa_host, dconudt_wetdep_host,
        dcondt_resusp_host, dcondt_prevap_host, dcondt_prevap_hist_host,
        fa_u_host, dpdry_i_host;
    Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>
        dconudt_activa_dev, dconudt_wetdep_dev, dcondt_resusp_dev,
        dcondt_prevap_dev, dcondt_prevap_hist_dev;
    mam4::ColumnView fa_u_dev, dpdry_i_dev;
    get_input(input, "dconudt_activa", nlevp, pcnst_extd, dconudt_activa_host,
              dconudt_activa_dev);
    get_input(input, "dconudt_wetdep", nlevp, pcnst_extd, dconudt_wetdep_host,
              dconudt_wetdep_dev);
    get_input(input, "dcondt_resusp", nlev, pcnst_extd, dcondt_resusp_host,
              dcondt_resusp_dev);
    get_input(input, "dcondt_prevap", nlev, pcnst_extd, dcondt_prevap_host,
              dcondt_prevap_dev);
    get_input(input, "dcondt_prevap_hist", nlev, pcnst_extd,
              dcondt_prevap_hist_host, dcondt_prevap_hist_dev);
    get_input(input, "fa_u", nlev, fa_u_host, fa_u_dev);
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);

    mam4::ColumnView sumactiva_dev =
        mam4::validation::create_column_view(pcnst_extd);
    mam4::ColumnView sumaqchem_dev =
        mam4::validation::create_column_view(pcnst_extd);
    mam4::ColumnView sumwetdep_dev =
        mam4::validation::create_column_view(pcnst_extd);
    mam4::ColumnView sumresusp_dev =
        mam4::validation::create_column_view(pcnst_extd);
    mam4::ColumnView sumprevap_dev =
        mam4::validation::create_column_view(pcnst_extd);
    mam4::ColumnView sumprevap_hist_dev =
        mam4::validation::create_column_view(pcnst_extd);
    std::vector<Real> sumactiva_host(pcnst_extd);
    std::vector<Real> sumaqchem_host(pcnst_extd);
    std::vector<Real> sumwetdep_host(pcnst_extd);
    std::vector<Real> sumresusp_host(pcnst_extd);
    std::vector<Real> sumprevap_host(pcnst_extd);
    std::vector<Real> sumprevap_hist_host(pcnst_extd);
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool doconvproc_extd[pcnst_extd] = {};
          for (int n = 0; n < pcnst_extd; ++n)
            doconvproc_extd[n] = doconvproc_extd_dev[n];
          const Real *dpdry_i = dpdry_i_dev.data();
          const Real *fa_u = fa_u_dev.data();
          Real *sumactiva = sumactiva_dev.data();
          Real *sumaqchem = sumaqchem_dev.data();
          Real *sumwetdep = sumwetdep_dev.data();
          Real *sumresusp = sumresusp_dev.data();
          Real *sumprevap = sumprevap_dev.data();
          Real *sumprevap_hist = sumprevap_hist_dev.data();
          mam4::convproc::compute_column_tendency(team,
              doconvproc_extd, ktop, kbot_prevap, dpdry_i, dcondt_resusp_dev,
              dcondt_prevap_dev, dcondt_prevap_hist_dev, dconudt_activa_dev,
              dconudt_wetdep_dev, fa_u, sumactiva, sumaqchem, sumwetdep,
              sumresusp, sumprevap, sumprevap_hist);
        });
    set_output(output, "sumactiva", pcnst_extd, sumactiva_host, sumactiva_dev);
    set_output(output, "sumaqchem", pcnst_extd, sumaqchem_host, sumaqchem_dev);
    set_output(output, "sumwetdep", pcnst_extd, sumwetdep_host, sumwetdep_dev);
    set_output(output, "sumresusp", pcnst_extd, sumresusp_host, sumresusp_dev);
    set_output(output, "sumprevap", pcnst_extd, sumprevap_host, sumprevap_dev);
    set_output(output, "sumprevap_hist", pcnst_extd, sumprevap_hist_host,
               sumprevap_hist_dev);
  });
}
