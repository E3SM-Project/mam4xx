// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <catch2/catch.hpp>

#include <mam4xx/aero_model.hpp>
#include <mam4xx/convproc.hpp>
#include <validation.hpp>

using namespace skywalker;

namespace {

void init_scratch(
    Kokkos::View<Real *>
        scratch1Dviews[mam4::ConvProc::Col1DViewInd::NumScratch]) {
  const mam4::ConvProc::Config config_;
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::q],
                 config_.nlev * mam4::aero_model::pcnst);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::mu], 1 + config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::md], 1 + config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::eudp], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dudp], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::eddp], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dddp], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::rhoair], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::zmagl], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::zmagl], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::gath],
                 (1 + config_.nlev) * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::chat],
                 (1 + config_.nlev) * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::conu],
                 (1 + config_.nlev) * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::cond],
                 (1 + config_.nlev) * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dconudt_wetdep],
                 (1 + config_.nlev) * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dconudt_activa],
                 (1 + config_.nlev) * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::fa_u], config_.nlev);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dcondt],
                 config_.nlev * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dcondt_wetdep],
                 config_.nlev * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dcondt_prevap],
                 config_.nlev * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dcondt_prevap_hist],
                 config_.nlev * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::dcondt_resusp],
                 config_.nlev * mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::wd_flux],
                 mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::sumactiva],
                 mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::sumaqchem],
                 mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::sumprevap],
                 mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::sumprevap_hist],
                 mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::sumresusp],
                 mam4::ConvProc::pcnst_extd);
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::sumwetdep],
                 mam4::ConvProc::pcnst_extd);
}

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
               Kokkos::View<Real * [mam4::aero_model::pcnst],
                            Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(col_view.data(), rows,
                                                       cols);
  EKAT_ASSERT(host.size() == rows * cols);
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
template <int COLS>
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Kokkos::View<Real *[COLS]> &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int j = 0, n = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i, ++n) {
      host[n] = host_view(i, j);
    }
  output.set(name, host);
}
} // namespace
void ma_convproc_tend(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();

  auto aero_species =
      mam4::aero_species_on_device(mam4::default_aero_species());

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int ktop = 47;
    const int kbot = 71;
    const Real dt = 36000;
    const mam4::ConvProc::convtype convtype = mam4::ConvProc::Deep;
    // const int pcnst_extd = mam4::ConvProc::pcnst_extd;
    const int pcnst = mam4::aero_model::pcnst;
    const int nsrflx = 6;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    // ktop is jt(il1g) to jt(il2g) in Fortran
    // but jt is just a scalar and il1g=il2g=48;
    // kbot is mx(il1g) to jt(il2g) in Fortran
    // but mx is just a scalar iand l1g=il2g=71;
    EKAT_ASSERT(input.get("jt") == 48);
    EKAT_ASSERT(input.get("mx") == 71);
    EKAT_ASSERT(input.get("il1g") == 1);
    EKAT_ASSERT(input.get("il2g") == 1);
    EKAT_ASSERT(input.get("dt") == 3600);
    EKAT_ASSERT(input.get("ncnst") == 40);
    EKAT_ASSERT(input.get("nsrflx") == 6);

    int mmtoo_prevap_resusp[pcnst];
    {
      const int resusp[pcnst] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                 0,  0,  0,  0,  0,  31, 33, 34, 32, 29,
                                 30, 35, -2, 31, 34, 30, 35, -2, 29, 30,
                                 31, 32, 33, 34, 35, -2, 33, 32, 35, -2};
      for (int i = 0; i < pcnst; ++i)
        mmtoo_prevap_resusp[i] = resusp[i] - 1;
    }

    std::vector<Real> temperature_host, pmid_host, qnew_host, mu_host, md_host,
        du_host, eu_host, ed_host, dp_host, dpdry_host, cldfrac_host,
        icwmr_host, rprd_host, evapc_host, doconvproc_host, species_class_host,
        dqdt_host, qsrflx_host;
    mam4::ColumnView temperature_dev, pmid_dev, mu_dev, md_dev, du_dev, eu_dev,
        ed_dev, dp_dev, dpdry_dev, cldfrac_dev, icwmr_dev, rprd_dev, evapc_dev,
        doconvproc_dev, species_class_dev;

    Kokkos::View<Real *[pcnst]> dqdt_dev("dqdt", nlev, pcnst);
    Kokkos::View<Real *[nsrflx]> qsrflx_dev("qsrflx", pcnst, nsrflx);
    Kokkos::View<Real *[pcnst], Kokkos::MemoryUnmanaged> qnew_dev;

    get_input(input, "temperature", nlev, temperature_host, temperature_dev);
    get_input(input, "pmid", nlev, pmid_host, pmid_dev);
    get_input(input, "qnew", nlev, pcnst, qnew_host, qnew_dev);
    get_input(input, "mu", nlev, mu_host, mu_dev);
    get_input(input, "md", nlev, md_host, md_dev);
    get_input(input, "du", nlev, du_host, du_dev);
    get_input(input, "eu", nlev, eu_host, eu_dev);
    get_input(input, "ed", nlev, ed_host, ed_dev);
    get_input(input, "dp", nlev, dp_host, dp_dev);
    get_input(input, "dpdry", nlev, dpdry_host, dpdry_dev);
    get_input(input, "cldfrac", nlev, cldfrac_host, cldfrac_dev);
    get_input(input, "icwmr", nlev, icwmr_host, icwmr_dev);
    get_input(input, "rprd", nlev, rprd_host, rprd_dev);
    get_input(input, "evapc", nlev, evapc_host, evapc_dev);
    get_input(input, "doconvproc", pcnst, doconvproc_host, doconvproc_dev);
    get_input(input, "species_class", pcnst, species_class_host,
              species_class_dev);

    Kokkos::View<Real *>
        scratch1Dviews[mam4::ConvProc::Col1DViewInd::NumScratch];
    init_scratch(scratch1Dviews);

    mam4::ColumnView scalars_dev = mam4::validation::create_column_view(3);
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Real cldfrac[nlev], icwmr[nlev], pmid[nlev], rprd[nlev], dpdry[nlev],
              evapc[nlev], du[nlev], eu[nlev], ed[nlev], dp[nlev],
              temperature[nlev], dqdt[nlev][pcnst], qsrflx[pcnst][nsrflx];
          int species_class[pcnst];
          bool doconvproc[pcnst];
          for (int i = 0; i < nlev; ++i) {
            cldfrac[i] = cldfrac_dev[i];
            icwmr[i] = icwmr_dev[i];
            temperature[i] = temperature_dev[i];
            pmid[i] = pmid_dev[i];
            rprd[i] = rprd_dev[i];
            dpdry[i] = dpdry_dev[i];
            evapc[i] = evapc_dev[i];
            du[i] = du_dev[i];
            eu[i] = eu_dev[i];
            ed[i] = ed_dev[i];
            dp[i] = dp_dev[i];
          }
          for (int i = 0; i < pcnst; ++i) {
            doconvproc[i] = doconvproc_dev[i];
            species_class[i] = species_class_dev[i];
          }
          auto dqdt_view = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
              &dqdt[0][0], nlev, pcnst);
          Real xx_mfup_max, xx_wcldbase;
          int xx_kcldbase;
          mam4::convproc::ma_convproc_tend(team,
              aero_species, scratch1Dviews, nlev, convtype, dt, temperature,
              pmid, qnew_dev, du, eu, ed, dp, dpdry, ktop, kbot,
              mmtoo_prevap_resusp, cldfrac, icwmr, rprd, evapc, dqdt_view,
              doconvproc, qsrflx, species_class, xx_mfup_max, xx_wcldbase,
              xx_kcldbase);

          for (int i = 0; i < nlev; ++i) {
            for (int j = 0; j < pcnst; ++j) {
              dqdt_dev(i, j) = dqdt_view(i, j);
            }
          }
          for (int i = 0; i < pcnst; ++i) {
            for (int j = 0; j < nsrflx; ++j) {
              qsrflx_dev(i, j) = qsrflx[i][j];
            }
          }
          scalars_dev[0] = xx_mfup_max;
          scalars_dev[1] = xx_wcldbase;
          scalars_dev[2] = xx_kcldbase;
        });
    set_output(output, "dqdt", nlev, pcnst, dqdt_host, dqdt_dev);
    set_output(output, "qsrflx", pcnst, nsrflx, qsrflx_host, qsrflx_dev);
    Real xx_mfup_max_host, xx_wcldbase_host;
    int xx_kcldbase_host;
    {
      auto host_view = Kokkos::create_mirror_view(scalars_dev);
      Kokkos::deep_copy(host_view, scalars_dev);
      xx_mfup_max_host = host_view[0];
      xx_wcldbase_host = host_view[1];
      xx_kcldbase_host = host_view[2];
    }
    // Make indexes Fortran based again.
    output.set("xx_mfup_max", xx_mfup_max_host);
    output.set("xx_wcldbase", xx_wcldbase_host);
    output.set("xx_kcldbase", xx_kcldbase_host + 1);
  });
}
