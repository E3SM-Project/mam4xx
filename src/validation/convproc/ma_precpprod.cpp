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
               Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
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
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void ma_precpprod(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int kk = input.get("kk") - 1;
    EKAT_ASSERT(53 == kk);
    const int pcnst_extd = input.get("pcnst_extd");
    EKAT_ASSERT(mam4::ConvProc::pcnst_extd == pcnst_extd);
    const Real x_ratio = input.get("x_ratio");
    Real pr_flux = input.get("pr_flux");
    Real pr_flux_tmp = input.get("pr_flux_tmp");
    Real pr_flux_base = input.get("pr_flux_base");

    std::vector<Real> rprd_host, dpdry_i_host, dcondt_wetdep_host,
        doconvproc_extd_host, species_class_host, wd_flux_host, dcondt_host,
        dcondt_prevap_host, dcondt_prevap_hist_host, mmtoo_prevap_resusp_host;
    mam4::ColumnView rprd_dev, dpdry_i_dev, doconvproc_extd_dev,
        species_class_dev, wd_flux_dev, mmtoo_prevap_resusp_dev;
    get_input(input, "rprd", nlev, rprd_host, rprd_dev);
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> dcondt_wetdep_dev,
        dcondt_dev, dcondt_prevap_dev, dcondt_prevap_hist_dev;
    get_input(input, "dcondt_wetdep", nlev, pcnst_extd, dcondt_wetdep_host,
              dcondt_wetdep_dev);
    get_input(input, "species_class", mam4::aero_model::pcnst,
              species_class_host, species_class_dev);
    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    get_input(input, "wd_flux", pcnst_extd, wd_flux_host, wd_flux_dev);
    get_input(input, "dcondt", nlev, pcnst_extd, dcondt_host, dcondt_dev);
    get_input(input, "dcondt_prevap", nlev, pcnst_extd, dcondt_prevap_host,
              dcondt_prevap_dev);
    get_input(input, "dcondt_prevap_hist", nlev, pcnst_extd,
              dcondt_prevap_hist_host, dcondt_prevap_hist_dev);
    get_input(input, "mmtoo_prevap_resusp", mam4::aero_model::pcnst,
              mmtoo_prevap_resusp_host, mmtoo_prevap_resusp_dev);
    mam4::ColumnView return_vals = mam4::validation::create_column_view(3);
    Kokkos::parallel_for(
        "ma_precpprod", 1, KOKKOS_LAMBDA(int) {
          Real rprd[nlev];
          for (int i = 0; i < nlev; ++i)
            rprd[i] = rprd_dev(i);
          Real dpdry_i[nlev];
          for (int i = 0; i < nlev; ++i)
            dpdry_i[i] = dpdry_i_dev(i);
          bool doconvproc_extd[mam4::ConvProc::pcnst_extd];
          for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          int species_class[mam4::aero_model::pcnst];
          for (int i = 0; i < mam4::aero_model::pcnst; ++i)
            species_class[i] = species_class_dev[i];
          int mmtoo_prevap_resusp[mam4::aero_model::pcnst];
          for (int i = 0; i < mam4::aero_model::pcnst; ++i)
            mmtoo_prevap_resusp[i] = mmtoo_prevap_resusp_dev[i] - 1;

          auto dcondt_wetdep =
              Kokkos::subview(dcondt_wetdep_dev, kk, Kokkos::ALL());
          auto dcondt = Kokkos::subview(dcondt_dev, kk, Kokkos::ALL());
          auto dcondt_prevap =
              Kokkos::subview(dcondt_prevap_dev, kk, Kokkos::ALL());
          auto dcondt_prevap_hist =
              Kokkos::subview(dcondt_prevap_hist_dev, kk, Kokkos::ALL());
          Real flux = pr_flux;
          Real flux_tmp = pr_flux_tmp;
          Real flux_base = pr_flux_base;
          // local precip flux [(kg/kg/s)*mb]
          const Real flux_local = mam4::max(0.0, rprd[kk] * dpdry_i[kk]);
          flux_base = mam4::max(0.0, flux_base + flux_local);
          flux =
              mam4::utils::min_max_bound(0.0, flux_base, flux_tmp + flux_local);
          mam4::convproc::ma_precpprod(
              rprd[kk], dpdry_i[kk], doconvproc_extd, x_ratio, species_class,
              mmtoo_prevap_resusp, wd_flux_dev, dcondt_wetdep, dcondt,
              dcondt_prevap, dcondt_prevap_hist);
          return_vals[0] = flux;
          return_vals[1] = flux_tmp;
          return_vals[2] = flux_base;
        });
    auto host_view = Kokkos::create_mirror_view(return_vals);
    Kokkos::deep_copy(host_view, return_vals);
    output.set("pr_flux", pr_flux);
    output.set("pr_flux_tmp", pr_flux_tmp);
    output.set("pr_flux_base", pr_flux_base);
    set_output(output, "wd_flux", pcnst_extd, wd_flux_host, wd_flux_dev);
    set_output(output, "dcondt", nlev, pcnst_extd, dcondt_host, dcondt_dev);
    set_output(output, "dcondt_prevap", nlev, pcnst_extd, dcondt_prevap_host,
               dcondt_prevap_dev);
    set_output(output, "dcondt_prevap_hist", nlev, pcnst_extd,
               dcondt_prevap_hist_host, dcondt_prevap_hist_dev);
  });

  // Check some corner cases.
  const int nlev = 72;
  mam4::ColumnView return_vals = mam4::validation::create_column_view(6);
  mam4::ColumnView dcondt_wetdep =
      mam4::validation::create_column_view(mam4::ConvProc::pcnst_extd);
  mam4::ColumnView dcondt =
      mam4::validation::create_column_view(mam4::ConvProc::pcnst_extd);
  mam4::ColumnView dcondt_prevap =
      mam4::validation::create_column_view(mam4::ConvProc::pcnst_extd);
  mam4::ColumnView dcondt_prevap_hist =
      mam4::validation::create_column_view(mam4::ConvProc::pcnst_extd);
  mam4::ColumnView wd_flux =
      mam4::validation::create_column_view(mam4::ConvProc::pcnst_extd);
  Kokkos::parallel_for(
      "ma_precpprod", 1, KOKKOS_LAMBDA(int) {
        Real rprd[nlev];
        for (int i = 0; i < nlev; ++i)
          rprd[i] = 1;
        Real dpdry_i[nlev];
        for (int i = 0; i < nlev; ++i)
          dpdry_i[i] = 1;
        bool doconvproc_extd[mam4::ConvProc::pcnst_extd];
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          doconvproc_extd[i] = 1;
        int species_class[mam4::aero_model::pcnst];
        for (int i = 0; i < mam4::aero_model::pcnst; ++i)
          species_class[i] = 1;
        int mmtoo_prevap_resusp[mam4::aero_model::pcnst];
        for (int i = 0; i < mam4::aero_model::pcnst; ++i)
          mmtoo_prevap_resusp[i] = 1;

        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          wd_flux[i] = 1;
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          dcondt_wetdep[i] = 1;
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          dcondt[i] = 1;
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          dcondt_prevap[i] = 1;
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          dcondt_prevap_hist[i] = 1;
        Real flux = 1;
        Real flux_tmp = 0;
        Real flux_base = 0;
        Real x_ratio = 0;
        int kk = 1;
        // local precip flux [(kg/kg/s)*mb]
        const Real flux_local = mam4::max(0.0, rprd[kk] * dpdry_i[kk]);
        flux_base = mam4::max(0.0, flux_base + flux_local);
        flux =
            mam4::utils::min_max_bound(0.0, flux_base, flux_tmp + flux_local);
        mam4::convproc::ma_precpprod(
            rprd[kk], dpdry_i[kk], doconvproc_extd, x_ratio, species_class,
            mmtoo_prevap_resusp, wd_flux, dcondt_wetdep, dcondt, dcondt_prevap,
            dcondt_prevap_hist);
        return_vals[0] = flux;
        return_vals[1] = flux_tmp;
        return_vals[2] = flux_base;
        return_vals[3] = 0;
        return_vals[4] = 0;
        return_vals[5] = 0;
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          return_vals[3] += dcondt[i];
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          return_vals[4] += dcondt_prevap[i];
        for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
          return_vals[5] += dcondt_prevap_hist[i];
      });
  auto host_view = Kokkos::create_mirror_view(return_vals);
  Kokkos::deep_copy(host_view, return_vals);
  Real pr_flux_base = host_view(0);
  Real pr_flux_tmp = host_view(1);
  Real x_ratio = host_view(2);
  Real x_dcondt = host_view(3);
  Real x_dcondt_prevap = host_view(4);
  Real x_dcondt_prevap_hist = host_view(5);
  EKAT_REQUIRE_MSG(pr_flux_base == 1, "Special case of input failed.");
  EKAT_REQUIRE_MSG(pr_flux_tmp == 0, "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_ratio == 1, "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_dcondt == 1 + 2 * (mam4::ConvProc::pcnst_extd - 1),
                   "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_dcondt_prevap == 1 + 2 * (mam4::ConvProc::pcnst_extd - 1),
                   "Special case of input failed.");
  EKAT_REQUIRE_MSG(x_dcondt_prevap_hist ==
                       1 + 2 * (mam4::ConvProc::pcnst_extd - 1),
                   "Special case of input failed.");
}
