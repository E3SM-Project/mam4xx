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
               Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
  EKAT_ASSERT(host.size() == rows * cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
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
                const Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void compute_wetdep_tend(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int pcnst_extd = mam4::ConvProc::pcnst_extd;
    const int nlev = 72;
    // Fetch ensemble parameters

    // delta t (model time increment) [s]
    const Real dt = input.get("dt");
    EKAT_ASSERT(dt == 3600);
    // Convert to C++ offset
    const int kk = input.get("kk") - 1;
    EKAT_ASSERT(kk == 53);

    // flag for doing convective transport
    std::vector<Real> doconvproc_extd_host, dt_u_host, dp_i_host,
        cldfrac_i_host, mu_p_eudp_host, aqfrac_host, icwmr_host, rprd_host,
        conu_host, dconudt_wetdep_host;
    mam4::ColumnView doconvproc_extd_dev, dt_u_dev, dp_i_dev, cldfrac_i_dev,
        mu_p_eudp_dev, aqfrac_dev, icwmr_dev, rprd_dev;
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> conu_dev, dconudt_wetdep_dev;

    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    get_input(input, "dt_u", nlev, dt_u_host, dt_u_dev);
    get_input(input, "dp_i", nlev, dp_i_host, dp_i_dev);
    get_input(input, "cldfrac_i", nlev, cldfrac_i_host, cldfrac_i_dev);
    get_input(input, "mu_p_eudp", nlev, mu_p_eudp_host, mu_p_eudp_dev);
    get_input(input, "aqfrac", pcnst_extd, aqfrac_host, aqfrac_dev);
    get_input(input, "icwmr", nlev, icwmr_host, icwmr_dev);
    get_input(input, "rprd", nlev, rprd_host, rprd_dev);
    get_input(input, "conu", nlev + 1, pcnst_extd, conu_host, conu_dev);
    get_input(input, "dconudt_wetdep", nlev + 1, pcnst_extd,
              dconudt_wetdep_host, dconudt_wetdep_dev);

    Kokkos::parallel_for(
        "compute_wetdep_tend", 1, KOKKOS_LAMBDA(int) {
          Real dt_u[nlev], dp_i[nlev], cldfrac_i[nlev], mu_p_eudp[nlev],
              icwmr[nlev], rprd[nlev];
          for (int n = 0; n < nlev; ++n) {
            dt_u[n] = dt_u_dev[n];
            dp_i[n] = dp_i_dev[n];
            cldfrac_i[n] = cldfrac_i_dev[n];
            mu_p_eudp[n] = mu_p_eudp_dev[n];
            icwmr[n] = icwmr_dev[n];
            rprd[n] = rprd_dev[n];
          }
          Real conu[nlev + 1][pcnst_extd], dconudt_wetdep[nlev + 1][pcnst_extd];
          for (int i = 0; i < nlev + 1; ++i) {
            for (int j = 0; j < pcnst_extd; ++j) {
              conu[i][j] = conu_dev(i, j);
              dconudt_wetdep[i][j] = dconudt_wetdep_dev(i, j);
            }
          }
          bool doconvproc_extd[pcnst_extd];
          Real aqfrac[pcnst_extd];
          for (int n = 0; n < pcnst_extd; ++n) {
            aqfrac[n] = aqfrac_dev[n];
            doconvproc_extd[n] = doconvproc_extd_dev[n];
          }

          mam4::convproc::compute_wetdep_tend(
              doconvproc_extd, /*disable_convective_incloud_wetdep=*/false, dt,
              dt_u[kk], dp_i[kk], cldfrac_i[kk], mu_p_eudp[kk], aqfrac,
              icwmr[kk], rprd[kk], conu[kk], dconudt_wetdep[kk]);
          for (int i = 0; i < nlev + 1; ++i) {
            for (int j = 0; j < pcnst_extd; ++j) {
              conu_dev(i, j) = conu[i][j];
              dconudt_wetdep_dev(i, j) = dconudt_wetdep[i][j];
            }
          }
        });

    set_output(output, "conu", nlev + 1, pcnst_extd, conu_host, conu_dev);
    set_output(output, "dconudt_wetdep", nlev + 1, pcnst_extd,
               dconudt_wetdep_host, dconudt_wetdep_dev);
  });
}
