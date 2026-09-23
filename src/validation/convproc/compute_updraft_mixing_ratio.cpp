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
               Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                            Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  EKAT_ASSERT(host.size() == rows * cols);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                     Kokkos::MemoryUnmanaged>(col_view.data(), rows, cols);
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
                std::vector<Real> &host, const mam4::ColumnView &dev) {
  host.resize(rows);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0; i < rows; ++i)
    host[i] = host_view(i);
  output.set(name, host);
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
void compute_updraft_mixing_ratio(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  auto aero_species =
      mam4::aero_species_on_device(mam4::default_aero_species());

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int pcnst_extd = mam4::ConvProc::pcnst_extd;
    // Fetch ensemble parameters
    const Real dt = input.get("dt");
    EKAT_ASSERT(std::abs(dt - 3600) < 1);
    const Real dz = input.get("dz");
    EKAT_ASSERT(std::abs(dz - 2799.6947) < 1);
    // Convert to C++ index by subtracting one.
    const int ktop = input.get("ktop") - 1;
    EKAT_ASSERT(47 == ktop);
    const int kbot = input.get("kbot");
    EKAT_ASSERT(71 == kbot);
    const int iconvtype = input.get("iconvtype");
    EKAT_ASSERT(1 == iconvtype);
    Real xx_wcldbase = input.get("xx_wcldbase");
    EKAT_ASSERT(0 == xx_wcldbase);
    int xx_kcldbase = input.get("xx_kcldbase");
    EKAT_ASSERT(0 == xx_kcldbase);

    std::vector<Real> doconvproc_extd_host, dp_i_host, dpdry_i_host,
        cldfrac_host, rhoair_i_host, zmagl_host, mu_i_host, eudp_host,
        gath_host, temperature_host, aqfrac_host, icwmr_host, rprd_host,
        fa_u_host, dconudt_wetdep_host, dconudt_activa_host, conu_host;

    mam4::ColumnView doconvproc_extd_dev, dp_i_dev, dpdry_i_dev, cldfrac_dev,
        rhoair_i_dev, zmagl_dev, mu_i_dev, eudp_dev, temperature_dev,
        aqfrac_dev, icwmr_dev, rprd_dev, fa_u_dev, scalars_dev;
    Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged> gath_dev,
        dconudt_wetdep_dev, dconudt_activa_dev, conu_dev;

    get_input(input, "doconvproc_extd", pcnst_extd, doconvproc_extd_host,
              doconvproc_extd_dev);
    get_input(input, "dp_i", nlev, dp_i_host, dp_i_dev);
    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    get_input(input, "cldfrac", nlev, cldfrac_host, cldfrac_dev);
    get_input(input, "rhoair_i", nlev, rhoair_i_host, rhoair_i_dev);
    get_input(input, "zmagl", nlev, zmagl_host, zmagl_dev);
    get_input(input, "mu_i", nlev + 1, mu_i_host, mu_i_dev);
    get_input(input, "eudp", nlev, eudp_host, eudp_dev);
    get_input(input, "temperature", nlev, temperature_host, temperature_dev);
    get_input(input, "aqfrac", pcnst_extd, aqfrac_host, aqfrac_dev);
    get_input(input, "icwmr", nlev, icwmr_host, icwmr_dev);
    get_input(input, "rprd", nlev, rprd_host, rprd_dev);

    get_input(input, "const", nlev, pcnst_extd, gath_host, gath_dev);
    get_input(input, "conu", nlev + 1, pcnst_extd, conu_host, conu_dev);

    fa_u_host.resize(nlev);
    fa_u_dev = mam4::validation::create_column_view(nlev);
    {
      dconudt_wetdep_host.resize((nlev + 1) * pcnst_extd);
      mam4::ColumnView dev =
          mam4::validation::create_column_view((nlev + 1) * pcnst_extd);
      dconudt_wetdep_dev =
          Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
              dev.data(), nlev + 1, pcnst_extd);
    }
    {
      dconudt_activa_host.resize((nlev + 1) * pcnst_extd);
      mam4::ColumnView dev =
          mam4::validation::create_column_view((nlev + 1) * pcnst_extd);
      dconudt_activa_dev =
          Kokkos::View<Real *[pcnst_extd], Kokkos::MemoryUnmanaged>(
              dev.data(), nlev + 1, pcnst_extd);
    }

    scalars_dev = mam4::validation::create_column_view(2);
    {
      auto host = Kokkos::create_mirror_view(scalars_dev);
      host(0) = xx_wcldbase;
      host(1) = xx_kcldbase;
      Kokkos::deep_copy(scalars_dev, host);
    }
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool doconvproc_extd[pcnst_extd];
          Real dp_i[nlev];
          Real dpdry_i[nlev];
          Real cldfrac[nlev];
          Real rhoair_i[nlev];
          Real zmagl[nlev];
          Real mu_i[nlev + 1];
          Real eudp[nlev + 1];
          Real temperature[nlev];
          Real aqfrac[pcnst_extd];
          Real icwmr[nlev];
          Real rprd[nlev];
          Real fa_u[nlev];

          for (int i = 0; i < pcnst_extd; ++i) {
            doconvproc_extd[i] = doconvproc_extd_dev[i];
            aqfrac[i] = aqfrac_dev[i];
          }
          for (int i = 0; i < nlev; ++i) {
            dp_i[i] = dp_i_dev(i);
            dpdry_i[i] = dpdry_i_dev(i);
            cldfrac[i] = cldfrac_dev(i);
            rhoair_i[i] = rhoair_i_dev(i);
            zmagl[i] = zmagl_dev(i);
            mu_i[i] = mu_i_dev(i);
            eudp[i] = eudp_dev(i);
            temperature[i] = temperature_dev(i);
            aqfrac[i] = aqfrac_dev(i);
            icwmr[i] = icwmr_dev(i);
            rprd[i] = rprd_dev(i);
            fa_u[i] = 0;
          }

          Real wcldbase = xx_wcldbase;
          int kcldbase = xx_kcldbase;
          mam4::convproc::compute_updraft_mixing_ratio(team,
              aero_species, doconvproc_extd, nlev, ktop, kbot, iconvtype, dt,
              dp_i, dpdry_i, cldfrac, rhoair_i, zmagl, dz, mu_i, eudp, gath_dev,
              temperature, aqfrac, icwmr, rprd, fa_u, dconudt_wetdep_dev,
              dconudt_activa_dev, conu_dev, wcldbase, kcldbase);
          scalars_dev(0) = wcldbase;
          scalars_dev(1) = kcldbase;
          for (int i = 0; i < nlev; ++i)
            fa_u_dev(i) = fa_u[i];
        });
    set_output(output, "dconudt_wetdep", nlev + 1, pcnst_extd,
               dconudt_wetdep_host, dconudt_wetdep_dev);
    set_output(output, "dconudt_activa", nlev + 1, pcnst_extd,
               dconudt_activa_host, dconudt_activa_dev);
    set_output(output, "fa_u", nlev, fa_u_host, fa_u_dev);
    {
      auto host = Kokkos::create_mirror_view(scalars_dev);
      Kokkos::deep_copy(host, scalars_dev);
      xx_wcldbase = host(0);
      // Convert back to Fortran indexing.
      xx_kcldbase = host(1) + 1;
    }
    output.set("xx_wcldbase", xx_wcldbase);
    output.set("xx_kcldbase", xx_kcldbase);
  });
}
