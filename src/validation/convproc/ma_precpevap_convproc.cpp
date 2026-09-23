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
void get_input(const int rows, const int cols, std::vector<Real> &host,
               Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                            Kokkos::MemoryUnmanaged> &dev) {
  host.resize(rows * cols, 0);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                     Kokkos::MemoryUnmanaged>(col_view.data(), rows, cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      host_view(i, j) = 0;
  Kokkos::deep_copy(dev, host_view);
}
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                                   Kokkos::MemoryUnmanaged> &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void ma_precpevap_convproc(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int ktop = input.get("ktop") - 1;
    EKAT_ASSERT(47 == ktop);
    const int pcnst_extd = input.get("pcnst_extd");
    EKAT_ASSERT(mam4::ConvProc::pcnst_extd == pcnst_extd);

    std::vector<Real> dcondt_host, dcondt_prevap_host, dcondt_prevap_hist_host,
        dcondt_wetdep_host, rprd_host, evapc_host, dpdry_i_host,
        doconvproc_extd_host, species_class_host, mmtoo_prevap_resusp_host;
    mam4::ColumnView rprd_dev, evapc_dev, dpdry_i_dev, doconvproc_extd_dev,
        species_class_dev, mmtoo_prevap_resusp_dev;
    Kokkos::View<Real * [mam4::ConvProc::pcnst_extd], Kokkos::MemoryUnmanaged>
        dcondt_dev, dcondt_prevap_dev, dcondt_prevap_hist_dev,
        dcondt_wetdep_dev;
    get_input(input, "rprd", nlev, rprd_host, rprd_dev);
    get_input(input, "evapc", nlev, evapc_host, evapc_dev);
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    get_input(input, "dcondt_wetdep", nlev, pcnst_extd, dcondt_wetdep_host,
              dcondt_wetdep_dev);
    get_input(nlev, pcnst_extd, dcondt_prevap_host, dcondt_prevap_dev);
    get_input(nlev, pcnst_extd, dcondt_prevap_hist_host,
              dcondt_prevap_hist_dev);

    get_input(input, "species_class", mam4::aero_model::pcnst,
              species_class_host, species_class_dev);
    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    get_input(input, "dcondt", nlev, pcnst_extd, dcondt_host, dcondt_dev);
    get_input(input, "mmtoo_prevap_resusp", mam4::aero_model::pcnst,
              mmtoo_prevap_resusp_host, mmtoo_prevap_resusp_dev);
    mam4::ColumnView wd_flux =
        mam4::validation::create_column_view(mam4::ConvProc::pcnst_extd);
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
     team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool doconvproc_extd[mam4::ConvProc::pcnst_extd];
          for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          int species_class[mam4::aero_model::pcnst];
          for (int i = 0; i < mam4::aero_model::pcnst; ++i)
            species_class[i] = species_class_dev[i];
          int mmtoo_prevap_resusp[mam4::aero_model::pcnst];
          for (int i = 0; i < mam4::aero_model::pcnst; ++i)
            mmtoo_prevap_resusp[i] = mmtoo_prevap_resusp_dev[i] - 1;

          mam4::convproc::ma_precpevap_convproc(team,
              ktop, nlev, dcondt_wetdep_dev, rprd_dev.data(), evapc_dev.data(),
              dpdry_i_dev.data(), doconvproc_extd, species_class,
              mmtoo_prevap_resusp, wd_flux, dcondt_prevap_dev,
              dcondt_prevap_hist_dev, dcondt_dev);
        });
    set_output(output, "dcondt", nlev, pcnst_extd, dcondt_host, dcondt_dev);
    set_output(output, "dcondt_prevap", nlev, pcnst_extd, dcondt_prevap_host,
               dcondt_prevap_dev);
    set_output(output, "dcondt_prevap_hist", nlev, pcnst_extd,
               dcondt_prevap_hist_host, dcondt_prevap_hist_dev);
  });
}
