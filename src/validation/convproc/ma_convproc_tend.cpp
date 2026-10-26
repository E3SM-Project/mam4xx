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

using View1D = typename mam4::DeviceType::view_1d<Real>;
using View2D = typename mam4::DeviceType::view_2d<Real>;
using View1DHost = typename mam4::HostType::view_1d<Real>;
using View2DHost = typename mam4::HostType::view_2d<Real>;
using ViewBool1D = typename mam4::DeviceType::view_1d<bool>;
using ViewInt1D = typename mam4::DeviceType::view_1d<int>;

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
  Kokkos::resize(scratch1Dviews[mam4::ConvProc::x_ratio], config_.nlev);
}

void get_input(const Input &input, const std::string &name, const int size,
               std::vector<Real> &host, View1D &dev) {
  host = input.get_array(name);
  dev = View1D(name, size);

  EKAT_ASSERT(host.size() == size);
  auto host_view = View1DHost((Real *)host.data(), size);
  Kokkos::deep_copy(dev, host_view);
}

void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, std::vector<Real> &host, View2D &dev) {
  host = input.get_array(name);
  dev = View2D(name, rows, cols);
  EKAT_ASSERT(host.size() == rows * cols);
  mam4::validation::convert_1d_vector_to_2d_view_device(host, dev);
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
  auto aero_species =
      mam4::aero_species_on_device(mam4::default_aero_species());

  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int nsrflx = 6;
    const mam4::ConvProc::convtype convtype = mam4::ConvProc::Deep;
    const int pcnst = mam4::aero_model::pcnst;

    // Fetch ensemble parameters as arrays
    auto jt_arr = input.get_array("jt");
    auto mx_arr = input.get_array("mx");
    auto dt_arr = input.get_array("dt");

    const int jt = std::round(jt_arr[0]);
    const int mx = std::round(mx_arr[0]);
    const Real dt = dt_arr[0];

    // Convert to C++ index by subtracting one
    const int ktop = jt - 1;
    const int kbot = mx;

    int mmtoo_prevap_resusp[pcnst];
    {
      const int resusp[pcnst] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                 0,  0,  0,  0,  0,  31, 33, 34, 32, 29,
                                 30, 35, -2, 31, 34, 30, 35, -2, 29, 30,
                                 31, 32, 33, 34, 35, -2, 33, 32, 35, -2};
      for (int i = 0; i < pcnst; ++i)
        mmtoo_prevap_resusp[i] = resusp[i] - 1;
    }

    // Fetch 1D input arrays
    std::vector<Real> temperature_host, pmid_host, mu_host, md_host,
        du_host, eu_host, ed_host, dp_host, dpdry_host, cldfrac_host,
        icwmr_host, rprd_host, evapc_host, doconvproc_host, species_class_host,
        dqdt_host, qsrflx_host;
    View1D temperature_dev, pmid_dev, mu_dev, md_dev, du_dev, eu_dev,
        ed_dev, dp_dev, dpdry_dev, cldfrac_dev, icwmr_dev, rprd_dev, evapc_dev;
    ViewBool1D doconvproc_dev("doconvproc", pcnst);
    ViewInt1D species_class_dev("species_class", pcnst);

    get_input(input, "temperature", nlev, temperature_host, temperature_dev);
    get_input(input, "pmid", nlev, pmid_host, pmid_dev);
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

    // Fetch 2D input arrays
    std::vector<Real> qnew_host;
    View2D qnew_dev;
    get_input(input, "qnew", nlev, pcnst, qnew_host, qnew_dev);

    // Fetch boolean/integer inputs
    {
      auto doconvproc_arr = input.get_array("doconvproc");
      auto doconvproc_host_view = Kokkos::create_mirror_view(doconvproc_dev);
      for (int i = 0; i < pcnst; ++i)
        doconvproc_host_view[i] = (bool)doconvproc_arr[i];
      Kokkos::deep_copy(doconvproc_dev, doconvproc_host_view);

      auto species_class_arr = input.get_array("species_class");
      auto species_class_host_view = Kokkos::create_mirror_view(species_class_dev);
      for (int i = 0; i < pcnst; ++i)
        species_class_host_view[i] = (int)species_class_arr[i];
      Kokkos::deep_copy(species_class_dev, species_class_host_view);
    }

    // Allocate output arrays on device
    Kokkos::View<Real *[pcnst]> dqdt_dev("dqdt", nlev, pcnst);
    Kokkos::View<Real *[nsrflx]> qsrflx_dev("qsrflx", pcnst, nsrflx);
    Kokkos::View<Real *> scalars_dev("scalars", 3);

    // Initialize output arrays to zero
    Kokkos::deep_copy(dqdt_dev, 0.0);
    Kokkos::deep_copy(qsrflx_dev, 0.0);
    Kokkos::deep_copy(scalars_dev, 0.0);

    // Initialize scratch arrays
    Kokkos::View<Real *>
        scratch1Dviews[mam4::ConvProc::Col1DViewInd::NumScratch];
    init_scratch(scratch1Dviews);

    // Run function in parallel_for for GPU compatibility
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Real cldfrac[nlev], icwmr[nlev], pmid[nlev], rprd[nlev], dpdry[nlev],
              evapc[nlev], du[nlev], eu[nlev], ed[nlev], dp[nlev],
              temperature[nlev], dqdt[nlev][pcnst], qsrflx[pcnst][nsrflx];
          int species_class[pcnst];
          bool doconvproc[pcnst];
          
          // Copy 1D arrays from device to stack
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
          
          // Copy boolean/integer arrays from device
          for (int i = 0; i < pcnst; ++i) {
            doconvproc[i] = doconvproc_dev[i];
            species_class[i] = species_class_dev[i];
          }
          
          // Initialize output arrays
          for (int i = 0; i < nlev; ++i)
            for (int j = 0; j < pcnst; ++j)
              dqdt[i][j] = 0.0;
          for (int i = 0; i < pcnst; ++i)
            for (int j = 0; j < nsrflx; ++j)
              qsrflx[i][j] = 0.0;
          
          // Create view from stack array for dqdt
          auto dqdt_view = Kokkos::View<Real **, Kokkos::MemoryUnmanaged>(
              &dqdt[0][0], nlev, pcnst);
          
          Real xx_mfup_max, xx_wcldbase;
          int xx_kcldbase;
          
          mam4::convproc::ma_convproc_tend(
              team, aero_species, scratch1Dviews, nlev, convtype, dt,
              temperature, pmid, qnew_dev, du, eu, ed, dp, dpdry, ktop, kbot,
              mmtoo_prevap_resusp, cldfrac, icwmr, rprd, evapc, dqdt_view,
              doconvproc, qsrflx, species_class, xx_mfup_max, xx_wcldbase,
              xx_kcldbase);

          // Copy results back to device
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
    Kokkos::fence();

    // Copy results back to host
    set_output(output, "dqdt", nlev, pcnst, dqdt_host, dqdt_dev);
    set_output(output, "qsrflx", pcnst, nsrflx, qsrflx_host, qsrflx_dev);

    Real xx_mfup_max_host, xx_wcldbase_host;
    int xx_kcldbase_host;
    {
      auto host_view = Kokkos::create_mirror_view(scalars_dev);
      Kokkos::deep_copy(host_view, scalars_dev);
      xx_mfup_max_host = host_view[0];
      xx_wcldbase_host = host_view[1];
      xx_kcldbase_host = (int)host_view[2];
    }

    // Make indexes Fortran based again
    output.set("xx_mfup_max", xx_mfup_max_host);
    output.set("xx_wcldbase", xx_wcldbase_host);
    output.set("xx_kcldbase", xx_kcldbase_host + 1);
  });
}
