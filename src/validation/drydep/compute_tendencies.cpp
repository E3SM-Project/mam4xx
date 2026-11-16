// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <aero_process.hpp>
#include <validation.hpp>

using namespace skywalker;

void compute_tendencies(Ensemble *ensemble) {

  mam4::AeroConfig mam4_config;

  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = mam4::nlev;
    const int aerosol_index = mam4::DryDeposition::index_interstitial_aerosols;

    EKAT_REQUIRE_MSG(input.has_array("ntot_amode"),
                     "Required name: ntot_amode");
    EKAT_REQUIRE_MSG(1 == input.get_array("ntot_amode").size(),
                     "ntot_amode array should be exactly 1 entry long.");
    EKAT_REQUIRE_MSG(4 == int(input.get_array("ntot_amode").front()),
                     "ntot_amode array should be exactly 4.");

    for (std::string s :
         {"tair", "pmid", "pint", "pdel", "obklen", "ustar", "landfrac",
          "icefrac", "ocnfrac", "fricvelin", "ram1in", "dt"})
      EKAT_REQUIRE_MSG(input.has_array(s), "Required name: " + s);

    for (std::string s : {"statq_", "qqcw_"}) {
      for (int i = aerosol_index; i < mam4::aero_model::pcnst; ++i) {
        const std::string name = s + std::to_string(i + 1);
        EKAT_REQUIRE_MSG(input.has_array(name), "Required name: " + name);
        EKAT_REQUIRE_MSG(nlev == input.get_array(name).size(),
                         name + " array should be exactly 72 entries long.");
      }
    }

    for (std::string s : {"dgncur_awet", "wetdens"}) {
      EKAT_REQUIRE_MSG(input.has_array(s), "Required name: " + s);
      EKAT_REQUIRE_MSG(nlev * mam4::AeroConfig::num_modes() ==
                           input.get_array(s).size(),
                       s + " array should be exactly 288 entry long.");
    }

    for (std::string s : {"tair", "pmid", "pdel"})
      EKAT_REQUIRE_MSG(nlev == input.get_array(s).size(),
                       s + " array should be exactly 72 entries long.");

    for (std::string s : {"pint"})
      EKAT_REQUIRE_MSG(1 + nlev == input.get_array(s).size(),
                       s + " array should be exactly 72 entries long.");

    for (std::string s : {"obklen", "ustar", "landfrac", "icefrac", "ocnfrac",
                          "fricvelin", "ram1in", "dt"})
      EKAT_REQUIRE_MSG(1 == input.get_array(s).size(),
                       s + " array should be exactly 1 entry long.");

    auto to_dev = [](const std::vector<Real> &vec) {
      mam4::ColumnView dev = mam4::validation::create_column_view(vec.size());
      auto host = Kokkos::create_mirror_view(dev);
      for (int i = 0; i < vec.size(); ++i)
        host[i] = vec[i];
      Kokkos::deep_copy(dev, host);
      return dev;
    };

    mam4::ColumnView tair = to_dev(input.get_array("tair"));
    mam4::ColumnView pmid = to_dev(input.get_array("pmid"));
    mam4::ColumnView pint = to_dev(input.get_array("pint"));
    mam4::ColumnView pdel = to_dev(input.get_array("pdel"));

    auto state_q_mem =
        mam4::validation::create_column_view(nlev * mam4::aero_model::pcnst);
    mam4::Diagnostics::ColumnTracerView state_q(state_q_mem.data(), nlev,
                                                mam4::aero_model::pcnst);
    {
      auto host_q = Kokkos::create_mirror_view(state_q);
      for (int i = aerosol_index; i < mam4::aero_model::pcnst; ++i) {
        const std::string name = "statq_" + std::to_string(i + 1);
        const std::vector<Real> vec = input.get_array(name);
        for (int j = 0; j < vec.size(); ++j)
          host_q(j, i) = vec[j];
      }
      Kokkos::deep_copy(state_q, host_q);
    }

    Kokkos::View<Real *> qqcw[mam4::aero_model::pcnst];
    for (int i = 0; i < mam4::aero_model::pcnst; ++i) {
      const std::string name = "qqcw_" + std::to_string(i + 1);
      if (i < aerosol_index)
        qqcw[i] = mam4::validation::create_column_view(nlev);
      else
        qqcw[i] = to_dev(input.get_array(name));
    }

    mam4::ColumnView dgncur_awet[mam4::AeroConfig::num_modes()];
    {
      std::vector<Real> dgncur_tot = input.get_array("dgncur_awet");
      for (int m = 0; m < mam4::AeroConfig::num_modes(); ++m) {
        std::vector<Real> dgncur(nlev);
        for (int lev = 0; lev < nlev; ++lev)
          dgncur[lev] = dgncur_tot[m * nlev + lev]; // col
        // dgncur[lev] = dgncur_tot[m + lev*AeroConfig::num_modes()]; // row
        dgncur_awet[m] = to_dev(dgncur);
      }
    }

    mam4::ColumnView wetdens[mam4::AeroConfig::num_modes()];
    {
      std::vector<Real> wetdens_tot = input.get_array("wetdens");
      for (int m = 0; m < mam4::AeroConfig::num_modes(); ++m) {
        std::vector<Real> wet(nlev);
        for (int lev = 0; lev < nlev; ++lev)
          wet[lev] = wetdens_tot[m * nlev + lev]; // col
        // wet[lev] = wetdens_tot[m + lev*AeroConfig::num_modes()]; // row
        wetdens[m] = to_dev(wet);
      }
    }

    const Real obklen = input.get_array("obklen").front();
    const Real ustar = input.get_array("ustar").front();
    const Real landfrac = input.get_array("landfrac").front();
    const Real icefrac = input.get_array("icefrac").front();
    const Real ocnfrac = input.get_array("ocnfrac").front();
    const Real fricvelin = input.get_array("fricvelin").front();
    const Real ram1in = input.get_array("ram1in").front();
    const Real dt = input.get_array("dt").front();

    auto ptend_q_mem =
        mam4::validation::create_column_view(nlev * mam4::aero_model::pcnst);
    mam4::Diagnostics::ColumnTracerView ptend_q(ptend_q_mem.data(), nlev,
                                                mam4::aero_model::pcnst);

    mam4::ColumnView aerdepdryis =
        mam4::validation::create_column_view(mam4::aero_model::pcnst);
    mam4::ColumnView aerdepdrycw =
        mam4::validation::create_column_view(mam4::aero_model::pcnst);

    mam4::ColumnView rho = mam4::validation::create_column_view(nlev);
    Kokkos::View<Real *> dqdt_tmp[mam4::aero_model::pcnst];
    for (int i = 0; i < mam4::aero_model::pcnst; ++i)
      Kokkos::resize(dqdt_tmp[i], nlev);

    Real pblh = 1000;
    mam4::Atmosphere atm = mam4::validation::create_atmosphere(nlev, pblh);
    mam4::Surface sfc = mam4::validation::create_surface();
    mam4::Prognostics progs = mam4::validation::create_prognostics(nlev);
    mam4::Diagnostics diags = mam4::validation::create_diagnostics(nlev);
    mam4::Tendencies tends = mam4::validation::create_tendencies(nlev);

    mam4::DryDepositionProcess::ProcessConfig process_config;
    mam4::DryDepositionProcess process(mam4_config, process_config);
    const Real t = 0;

    atm.temperature = tair;
    atm.pressure = pmid;
    atm.interface_pressure = pint;
    atm.hydrostatic_dp = pdel;

    diags.tracer_mixing_ratio = state_q;
    for (int i = 0; i < mam4::AeroConfig::num_modes(); ++i) {
      diags.wet_geometric_mean_diameter_i[i] = dgncur_awet[i];
      diags.wet_density[i] = wetdens[i];
    }
    diags.Obukhov_length = obklen;
    diags.surface_friction_velocty = ustar;
    diags.land_fraction = landfrac;
    diags.ice_fraction = icefrac;
    diags.ocean_fraction = ocnfrac;
    diags.friction_velocity = fricvelin;
    diags.aerodynamical_resistance = ram1in;
    diags.d_tracer_mixing_ratio_dt = ptend_q;
    diags.deposition_flux_of_cloud_borne_aerosols = aerdepdrycw;
    diags.deposition_flux_of_interstitial_aerosols = aerdepdryis;

    static constexpr int ntot_amode = mam4::AeroConfig::num_modes();
    const int num_aerosol = mam4::AeroConfig::num_aerosol_ids();
    for (int m = 0; m < ntot_amode; ++m) {
      progs.n_mode_c[m] = qqcw[mam4::ConvProc::numptrcw_amode(m)];
      for (int a = 0; a < num_aerosol; ++a)
        if (-1 < mam4::ConvProc::lmassptrcw_amode(a, m))
          progs.q_aero_c[m][a] = qqcw[mam4::ConvProc::lmassptrcw_amode(a, m)];
    }

    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          process.compute_tendencies(team, t, dt, atm, sfc, progs, diags,
                                     tends);
        });

    for (int m = 0; m < ntot_amode; ++m) {
      qqcw[mam4::ConvProc::numptrcw_amode(m)] = tends.n_mode_c[m] =
          tends.n_mode_c[m];
      for (int a = 0; a < num_aerosol; ++a)
        if (-1 < mam4::ConvProc::lmassptrcw_amode(a, m))
          qqcw[mam4::ConvProc::lmassptrcw_amode(a, m)] = tends.q_aero_c[m][a];
    }
    aerdepdrycw = diags.deposition_flux_of_cloud_borne_aerosols;
    aerdepdryis = diags.deposition_flux_of_interstitial_aerosols;
    ptend_q = diags.d_tracer_mixing_ratio_dt;
    Kokkos::fence();
    auto to_host = [](mam4::ColumnView dev) {
      auto host = Kokkos::create_mirror_view(dev);
      Kokkos::deep_copy(host, dev);
      std::vector<Real> vec(host.size());
      for (int i = 0; i < vec.size(); ++i)
        vec[i] = host[i];
      return vec;
    };
    auto ptend_host = Kokkos::create_mirror_view(ptend_q);
    std::vector<Real> is_host = to_host(aerdepdryis);
    std::vector<Real> cw_host = to_host(aerdepdrycw);
    Kokkos::deep_copy(ptend_host, ptend_q);
    std::vector<Real> ptend(nlev);
    for (int m = aerosol_index; m < mam4::aero_model::pcnst; ++m) {
      for (int lev = 0; lev < nlev; ++lev)
        ptend[lev] = ptend_host(lev, m);
      output.set("ptendq_" + std::to_string(m + 1), ptend);
      output.set("qqcw_" + std::to_string(m + 1), to_host(qqcw[m]));
      output.set("aerdepdryis_" + std::to_string(m + 1), is_host[m]);
      output.set("aerdepdrycw_" + std::to_string(m + 1), cw_host[m]);
    }
  });
}
