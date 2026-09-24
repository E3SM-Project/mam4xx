// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void aero_model_wetdep(Ensemble *ensemble) {
  mam4::AeroConfig aero_config;

  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename mam4::HostType::view_1d<Real>;
    using View2DHost = typename mam4::HostType::view_2d<Real>;
    using View2D = mam4::DeviceType::view_2d<Real>;
    using ConstView1D = mam4::DeviceType::view_1d<const Real>;

    mam4::Prognostics progs = mam4::validation::create_prognostics(mam4::nlev);
    mam4::Tendencies tends = mam4::validation::create_tendencies(mam4::nlev);
    int nlev = mam4::nlev;
    Real pblh = 1000;
    const Real dt = input.get_array("dt")[0];
    //
    View2D state_q("state_q", nlev, mam4::aero_model::pcnst);
    const auto state_q_db = input.get_array("state_q");
    mam4::validation::convert_1d_vector_to_2d_view_device(state_q_db, state_q);
    auto qqcw_db = input.get_array("qqcw"); // 2d

    View2D qqcw("qqcw", nlev, mam4::aero_model::pcnst);
    auto qqcw_host = Kokkos::create_mirror_view(qqcw);
    int count = 0;
    for (int kk = 0; kk < nlev; ++kk) {
      for (int i = 0; i < mam4::aero_model::pcnst; ++i) {
        qqcw_host(kk, i) = qqcw_db[count];
        count++;
      }
    }
    Kokkos::deep_copy(qqcw, qqcw_host);

    mam4::ColumnView temperature =
        mam4::validation::get_input_in_columnview(input, "temperature");
    mam4::ColumnView pressure =
        mam4::validation::get_input_in_columnview(input, "pmid");
    mam4::ColumnView hydrostatic_dp =
        mam4::validation::get_input_in_columnview(input, "pdel");

    auto vapor_mixing_ratio = mam4::testing::create_column_view(nlev);
    auto liquid_mixing_ratio = mam4::testing::create_column_view(nlev); //
    auto ice_mixing_ratio = mam4::testing::create_column_view(nlev);    //
    auto cloud_liquid_number_mixing_ratio =
        mam4::testing::create_column_view(nlev);
    auto cloud_ice_number_mixing_ratio =
        mam4::testing::create_column_view(nlev);
    // We need deep_copy because of executation error due to different layout
    // q[0] = atm.vapor_mixing_ratio(klev);               // qv
    Kokkos::deep_copy(vapor_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 0));
    // q[1] = atm.liquid_mixing_ratio(klev);              // qc
    Kokkos::deep_copy(liquid_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 1));
    // q[2] = atm.ice_mixing_ratio(klev);                 // qi
    Kokkos::deep_copy(ice_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 2));
    // q[3] = atm.cloud_liquid_number_mixing_ratio(klev); //  nc
    Kokkos::deep_copy(cloud_liquid_number_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 3));
    // q[4] = atm.cloud_ice_number_mixing_ratio(klev);    // ni
    Kokkos::deep_copy(cloud_ice_number_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 4));

    auto height = mam4::testing::create_column_view(nlev);
    auto interface_pressure = mam4::testing::create_column_view(nlev + 1);
    auto cloud_fraction = mam4::testing::create_column_view(nlev);
    auto updraft_vel_ice_nucleation = mam4::testing::create_column_view(nlev);

    auto atm = mam4::Atmosphere(
        nlev, temperature, pressure, vapor_mixing_ratio, liquid_mixing_ratio,
        cloud_liquid_number_mixing_ratio, ice_mixing_ratio,
        cloud_ice_number_mixing_ratio, height, hydrostatic_dp,
        interface_pressure, cloud_fraction, updraft_vel_ice_nucleation, pblh);

    auto prain =
        mam4::validation::get_input_in_columnview(input, "inputs_prain");
    // inputs
    mam4::ColumnView cldt =
        mam4::validation::get_input_in_columnview(input, "inputs_cldt");
    // Note that itim and itim_old are used separately for the cld variables
    // although they are the same, as also indicated by the discussion on the
    // Confluence page
    mam4::ColumnView cldn_prev_step =
        mam4::validation::get_input_in_columnview(input, "cldn"); // d

    mam4::ColumnView rprdsh =
        mam4::validation::get_input_in_columnview(input, "rprdsh");
    mam4::ColumnView rprddp =
        mam4::validation::get_input_in_columnview(input, "rprddp");
    mam4::ColumnView evapcdp =
        mam4::validation::get_input_in_columnview(input, "evapcdp");
    mam4::ColumnView evapcsh =
        mam4::validation::get_input_in_columnview(input, "evapcsh");

    mam4::ColumnView dp_frac =
        mam4::validation::get_input_in_columnview(input, "p_dp_frac");
    mam4::ColumnView sh_frac =
        mam4::validation::get_input_in_columnview(input, "p_sh_frac");
    mam4::ColumnView icwmrdp =
        mam4::validation::get_input_in_columnview(input, "p_icwmrdp");
    mam4::ColumnView icwmrsh =
        mam4::validation::get_input_in_columnview(input, "p_icwmrsh");

    mam4::ColumnView evapr =
        mam4::validation::get_input_in_columnview(input, "inputs_evapr"); //

    // outputs
    mam4::ColumnView dlf =
        mam4::validation::get_input_in_columnview(input, "dlf"); //
    mam4::wetdep::View1D aerdepwetcw("aerdepwetcw", mam4::aero_model::pcnst);
    mam4::wetdep::View1D aerdepwetis("aerdepwetis", mam4::aero_model::pcnst);
    const int num_modes = aero_config.num_modes();

    Kokkos::View<int *> isprx("isprx", nlev);

    View2DHost scavimptblvol_host("scavimptblvol_host",
                                  mam4::aero_model::nimptblgrow_total,
                                  aero_config.num_modes());
    View2DHost scavimptblnum_host("scavimptblnum_host",
                                  mam4::aero_model::nimptblgrow_total,
                                  aero_config.num_modes());

    mam4::wetdep::init_scavimptbl(aero_config, scavimptblvol_host,
                                  scavimptblnum_host);

    View2D scavimptblnum("scavimptblnum", mam4::aero_model::nimptblgrow_total,
                         aero_config.num_modes());
    View2D scavimptblvol("scavimptblvol", mam4::aero_model::nimptblgrow_total,
                         aero_config.num_modes());
    Kokkos::deep_copy(scavimptblnum, scavimptblnum_host);
    Kokkos::deep_copy(scavimptblvol, scavimptblvol_host);

    mam4::wetdep::View2D wet_geometric_mean_diameter_i(
        "wet_geometric_mean_diameter_i", num_modes, nlev);
    const auto dgnumwet_db = input.get_array("dgnumwet");
    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(
        dgnumwet_db, wet_geometric_mean_diameter_i);

    mam4::wetdep::View2D dry_geometric_mean_diameter_i(
        "dry_geometric_mean_diameter_i", num_modes, nlev);
    const auto dgncur_a_db = input.get_array("dgncur_a");
    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(
        dgncur_a_db, dry_geometric_mean_diameter_i);

    mam4::wetdep::View2D qaerwat("qaerwat", num_modes, nlev);
    const auto qaerwat_db = input.get_array("qaerwat");
    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(qaerwat_db,
                                                                    qaerwat);

    mam4::wetdep::View2D wetdens("wetdens", num_modes, nlev);
    const auto wetdens_db = input.get_array("wetdens");
    mam4::validation::convert_1d_vector_to_transpose_2d_view_device(wetdens_db,
                                                                    wetdens);

    mam4::wetdep::View2D ptend_q("ptend_q", nlev, mam4::aero_model::pcnst);

    // work arrays
    const int work_len = mam4::wetdep::get_aero_model_wetdep_work_len();
    mam4::wetdep::View1D work("work", work_len);

    mam4::modal_aero_calcsize::CalcsizeData cal_data;
    cal_data.initialize();
    const bool update_mmr = true;
    cal_data.set_update_mmr(update_mmr);

    ConstView1D mu_icol, md_icol, eu_icol, du_icol, ed_icol;
    ConstView1D dp_icol, p_del_dry_icol;
    ConstView1D dlfsh;           
    ConstView1D sh_e_ed_ratio;
    const bool convproc_do_aer=false, convproc_do_gas=false;
    const int ktop = 0;
    const int kbot = 0;
    const int* species_class = nullptr;
    const int* mmtoo_prevap_resusp = nullptr;

    auto team_policy = mam4::ThreadTeamPolicy(1u, mam4::testing::team_size);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Kokkos::View<Real*> scratch1Dviews[mam4::ConvProc::Col1DViewInd::NumScratch];
          auto progs_in = progs;
          auto tends_in = tends;

          // we need to inject validation values to progs.
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, nlev), [&](int kk) {
                // copy data from prog to stateq
                const auto state_q_kk = ekat::subview(state_q, kk);
                const auto qqcw_kk = ekat::subview(qqcw, kk);
                mam4::utils::inject_qqcw_to_prognostics(qqcw_kk, progs_in, kk);
                mam4::utils::inject_stateq_to_prognostics(state_q_kk, progs_in,
                                                          kk);
              });
          team.team_barrier();

          const Real scav_fraction_in_cloud_strat = 1.00;
          const Real scav_fraction_in_cloud_conv = 0.00;
          const Real scav_fraction_below_cloud_strat = 0.03;
          const Real activation_fraction_in_cloud_conv = 0.40;
          mam4::wetdep::aero_model_wetdep(
              team, atm, progs_in, tends_in, dt, scav_fraction_in_cloud_strat,
              scav_fraction_in_cloud_conv, scav_fraction_below_cloud_strat,
              activation_fraction_in_cloud_conv,
              // inputs
              cldt, rprdsh, rprddp, evapcdp, evapcsh, dp_frac, sh_frac, icwmrdp,
              icwmrsh, evapr,
              // outputs
              dlf, prain, scavimptblnum, scavimptblvol, cal_data,
              wet_geometric_mean_diameter_i, dry_geometric_mean_diameter_i,
              qaerwat, wetdens,
              // output
              aerdepwetis, aerdepwetcw, work, isprx,
              // Convection mass flux parameters
              scratch1Dviews,
              mu_icol, md_icol, du_icol, eu_icol, ed_icol,
              dp_icol, p_del_dry_icol, dlfsh, sh_e_ed_ratio,
              ktop, kbot,
              convproc_do_aer, convproc_do_gas,
              species_class, mmtoo_prevap_resusp,
              aero_config);

          team.team_barrier();
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, 0, nlev), [&](int kk) {
                const auto ptend_q_kk = ekat::subview(ptend_q, kk);
                mam4::utils::extract_ptend_from_tendencies(tends_in, ptend_q_kk,
                                                           kk);
              });
        });

    std::vector<Real> output_pcnst(mam4::aero_model::pcnst, 0);
    auto aerdepwetcw_host =
        View1DHost((Real *)output_pcnst.data(), mam4::aero_model::pcnst);
    Kokkos::deep_copy(aerdepwetcw_host, aerdepwetcw);
    output.set("aerdepwetcw", output_pcnst);

    std::vector<Real> aerdepwetis_output(mam4::aero_model::pcnst, 0);
    auto aerdepwetis_host =
        View1DHost((Real *)aerdepwetis_output.data(), mam4::aero_model::pcnst);
    Kokkos::deep_copy(aerdepwetis_host, aerdepwetis);
    output.set("aerdepwetis", aerdepwetis_output);

    std::vector<Real> output_modes(nlev * num_modes, 0);
    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        wet_geometric_mean_diameter_i, output_modes);
    output.set("dgnumwet", output_modes);

    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        dry_geometric_mean_diameter_i, output_modes);
    output.set("dgncur_a", output_modes);

    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        qaerwat, output_modes);
    output.set("qaerwat", output_modes);

    mam4::validation::convert_transpose_2d_view_device_to_1d_vector(
        wetdens, output_modes);
    output.set("wetdens", output_modes);
    // Note: Fortran validation code uses -9999.9 for ptend that are not
    // aerosols.
    using range_type = Kokkos::pair<int, int>;
    const auto &ptend_q_non = Kokkos::subview(
        ptend_q, Kokkos::ALL, range_type(0, mam4::utils::aero_start_ind()));
    Kokkos::deep_copy(ptend_q_non, -9999.900390625);
    std::vector<Real> output_ptend(nlev * mam4::aero_model::pcnst, 0);
    mam4::validation::convert_2d_view_device_to_1d_vector(ptend_q,
                                                          output_ptend);
    output.set("ptend_lq", output_ptend);
  });
}
