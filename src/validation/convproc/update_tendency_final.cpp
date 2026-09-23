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

void set_output(Output &output, const std::string &name, const int size,
                std::vector<Real> &host, const mam4::ColumnView &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int n = 0; n < size; ++n)
    host[n] = host_view[n];
  output.set(name, host);
}

void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, const bool col_major, std::vector<Real> &host,
               Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
  EKAT_ASSERT(host.size() == rows * cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    if (col_major)
      for (int j = 0, n = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i, ++n)
          matrix[i][j] = host[n];
    else
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
                const int cols, const bool col_major, std::vector<Real> &host,
                const Kokkos::View<Real **, Kokkos::MemoryUnmanaged> &dev) {
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  if (col_major)
    for (int j = 0, n = 0; j < cols; ++j)
      for (int i = 0; i < rows; ++i, ++n)
        host[n] = host_view(i, j);
  else
    for (int i = 0, n = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j, ++n)
        host[n] = host_view(i, j);
  output.set(name, host);
}
} // namespace
void update_tendency_final(Ensemble *ensemble) {

  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int pcnst = mam4::aero_model::pcnst;
    // Fetch ensemble parameters

    // delta t (model time increment) [s]
    const Real dt = input.get("dt");
    // these variables depend on mode No and k
    const int ktop = input.get("ktop");
    const int kbot_prevap = input.get("kbot_prevap");
    EKAT_ASSERT(72 == kbot_prevap);
    // number of sub timesteps
    const int ntsub = input.get("ntsub");
    // index of sub timesteps from the outer loop
    const int jtsub = input.get("jtsub");
    // number of tracers to transport
    const int ncnst = input.get("ncnst");
    EKAT_ASSERT(ncnst == pcnst);
    const int nsrflx = 6;

    // flag for doing convective transport
    std::vector<Real> doconvproc_host;
    mam4::ColumnView doconvproc_dev;
    get_input(input, "doconvproc", pcnst, doconvproc_host, doconvproc_dev);

    std::vector<Real> sumactiva_host, sumaqchem_host, sumwetdep_host,
        sumresusp_host, sumprevap_host, sumprevap_hist_host;
    mam4::ColumnView sumactiva_dev, sumaqchem_dev, sumwetdep_dev, sumresusp_dev,
        sumprevap_dev, sumprevap_hist_dev;
    get_input(input, "sumactiva", 2 * ncnst, sumactiva_host, sumactiva_dev);
    get_input(input, "sumaqchem", 2 * ncnst, sumaqchem_host, sumaqchem_dev);
    get_input(input, "sumwetdep", 2 * ncnst, sumwetdep_host, sumwetdep_dev);
    get_input(input, "sumresusp", 2 * ncnst, sumresusp_host, sumresusp_dev);
    get_input(input, "sumprevap", 2 * ncnst, sumprevap_host, sumprevap_dev);
    get_input(input, "sumprevap_hist", 2 * ncnst, sumprevap_hist_host,
              sumprevap_hist_dev);

    // grid-average TMR tendency for current column  [kg/kg/s]
    // JRO: Never could figure out why some 2D arrays are row major and other
    // col major.
    const bool col_major = true;
    const bool row_major = false;
    std::vector<Real> dcondt_host, dqdt_host, q_i_host, qsrflx_host;
    Kokkos::View<Real **, Kokkos::MemoryUnmanaged> dcondt_dev, dqdt_dev,
        q_i_dev, qsrflx_dev;

    get_input(input, "dcondt", 2 * ncnst, kbot_prevap, col_major, dcondt_host,
              dcondt_dev);
    get_input(input, "dqdt", ncnst, kbot_prevap, row_major, dqdt_host,
              dqdt_dev);
    get_input(input, "q_i", ncnst, kbot_prevap, row_major, q_i_host, q_i_dev);
    get_input(input, "qsrflx", ncnst, nsrflx, col_major, qsrflx_host,
              qsrflx_dev);

    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool doconvproc[pcnst] = {};
          for (int n = 0; n < pcnst; ++n)
            doconvproc[n] = doconvproc_dev[n];

          Real *sumactiva = sumactiva_dev.data();
          Real *sumaqchem = sumaqchem_dev.data();
          Real *sumwetdep = sumwetdep_dev.data();
          Real *sumresusp = sumresusp_dev.data();
          Real *sumprevap = sumprevap_dev.data();
          Real *sumprevap_hist = sumprevap_hist_dev.data();
          Real qsrflx[pcnst][nsrflx];
          for (int i = 0; i < ncnst; ++i)
            for (int j = 0; j < nsrflx; ++j)
              qsrflx[i][j] = qsrflx_dev(i, j);
          mam4::convproc::update_tendency_diagnostics(team,
              ntsub, ncnst, doconvproc, sumactiva, sumaqchem, sumwetdep,
              sumresusp, sumprevap, sumprevap_hist, qsrflx);
          for (int i = 0; i < ncnst; ++i)
            for (int j = 0; j < nsrflx; ++j)
              qsrflx_dev(i, j) = qsrflx[i][j];
        });

    Kokkos::parallel_for(
        "update_tendency_final", kbot_prevap, KOKKOS_LAMBDA(int klev) {
          if (ktop - 1 <= klev) {

            bool doconvproc[pcnst] = {};
            for (int n = 0; n < pcnst; ++n)
              doconvproc[n] = doconvproc_dev[n];

            Real dcondt[2 * pcnst];
            for (int i = 0; i < 2 * ncnst; ++i)
              dcondt[i] = dcondt_dev(i, klev);

            Real dqdt[pcnst];
            for (int i = 0; i < ncnst; ++i)
              dqdt[i] = dqdt_dev(i, klev);

            Real q_i[pcnst];
            for (int i = 0; i < ncnst; ++i)
              q_i[i] = q_i_dev(i, klev);

            mam4::convproc::update_tendency_final(
                ntsub, jtsub, ncnst, dt, dcondt, doconvproc, dqdt, q_i);
            for (int i = 0; i < ncnst; ++i)
              dqdt_dev(i, klev) = dqdt[i];
            for (int i = 0; i < ncnst; ++i)
              q_i_dev(i, klev) = q_i[i];
            for (int i = 0; i < 2 * ncnst; ++i)
              dcondt_dev(i, klev) = dcondt[i];
          }
        });

    set_output(output, "dqdt", ncnst, kbot_prevap, row_major, dqdt_host,
               dqdt_dev);
    set_output(output, "q_i", ncnst, kbot_prevap, row_major, q_i_host, q_i_dev);
    set_output(output, "qsrflx", ncnst, nsrflx, col_major, qsrflx_host,
               qsrflx_dev);
    set_output(output, "doconvproc", pcnst, doconvproc_host, doconvproc_dev);
    set_output(output, "sumactiva", 2 * ncnst, sumactiva_host, sumactiva_dev);
    set_output(output, "sumaqchem", 2 * ncnst, sumaqchem_host, sumaqchem_dev);
    set_output(output, "sumwetdep", 2 * ncnst, sumwetdep_host, sumwetdep_dev);
    set_output(output, "sumresusp", 2 * ncnst, sumresusp_host, sumresusp_dev);
    set_output(output, "sumprevap", 2 * ncnst, sumprevap_host, sumprevap_dev);
    set_output(output, "sumprevap_hist", 2 * ncnst, sumprevap_hist_host,
               sumprevap_hist_dev);
  });
}
