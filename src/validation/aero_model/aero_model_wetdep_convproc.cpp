// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/convproc.hpp>
#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;

void aero_model_wetdep_convproc(Ensemble *ensemble) {
  mam4::AeroConfig aero_config;

  ensemble->process([=](const Input &input, Output &output) {
    using View1DHost = typename mam4::HostType::view_1d<Real>;
    using View2DHost = typename mam4::HostType::view_2d<Real>;
    using View1D = mam4::DeviceType::view_1d<Real>;
    using View2D = mam4::DeviceType::view_2d<Real>;
    using ConstView1D = mam4::DeviceType::view_1d<const Real>;

    mam4::Prognostics progs = mam4::validation::create_prognostics(mam4::nlev);
    mam4::Tendencies tends = mam4::validation::create_tendencies(mam4::nlev);
    int nlev = mam4::nlev;
    Real pblh = 1000;
    const Real dt = input.get_array("dt")[0];

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
    auto liquid_mixing_ratio = mam4::testing::create_column_view(nlev);
    auto ice_mixing_ratio = mam4::testing::create_column_view(nlev);
    auto cloud_liquid_number_mixing_ratio =
        mam4::testing::create_column_view(nlev);
    auto cloud_ice_number_mixing_ratio =
        mam4::testing::create_column_view(nlev);

    Kokkos::deep_copy(vapor_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 0));
    Kokkos::deep_copy(liquid_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 1));
    Kokkos::deep_copy(ice_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 2));
    Kokkos::deep_copy(cloud_liquid_number_mixing_ratio,
                      Kokkos::subview(state_q, Kokkos::ALL(), 3));
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
    mam4::ColumnView cldn_prev_step =
        mam4::validation::get_input_in_columnview(input, "cldn");

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
        mam4::validation::get_input_in_columnview(input, "inputs_evapr");

    // Convective processing inputs
    mam4::ColumnView mu_icol =
        mam4::validation::get_input_in_columnview(input, "mu");
    mam4::ColumnView md_icol =
        mam4::validation::get_input_in_columnview(input, "md");
    mam4::ColumnView du_icol =
        mam4::validation::get_input_in_columnview(input, "du");
    mam4::ColumnView eu_icol =
        mam4::validation::get_input_in_columnview(input, "eu");
    mam4::ColumnView ed_icol =
        mam4::validation::get_input_in_columnview(input, "ed");

    // Pressure arrays for convective processing
    mam4::ColumnView dp_icol =
        mam4::validation::get_input_in_columnview(input, "dp");

    // Dry pressure thickness (pdeldry in Pa) converted to mb for dpdry in C++
    View1D p_del_dry_icol("p_del_dry_icol", nlev);
    if (input.has("pdeldry")) {
      const auto pdeldry_vec = input.get_array("pdeldry");
      View1DHost pdeldry_host((Real *)pdeldry_vec.data(), nlev);
      Kokkos::deep_copy(p_del_dry_icol, pdeldry_host);
    } else if (input.has("state_pdeldry")) {
      const auto pdeldry_vec = input.get_array("state_pdeldry");
      View1DHost pdeldry_host((Real *)pdeldry_vec.data(), nlev);
      Kokkos::deep_copy(p_del_dry_icol, pdeldry_host);
    } else {
      // Fall back to hydrostatic_dp if dry pressure is not separately provided
      Kokkos::deep_copy(p_del_dry_icol, hydrostatic_dp);
    }

    // Convert Pa to mb for convective dpdry
    auto p_del_dry_host = Kokkos::create_mirror_view(p_del_dry_icol);
    Kokkos::deep_copy(p_del_dry_host, p_del_dry_icol);
    for (int k = 0; k < nlev; ++k) {
      p_del_dry_host(k) /= 100.0;
    }
    Kokkos::deep_copy(p_del_dry_icol, p_del_dry_host);

    // Convective layer top and bottom indices
    // Fortran jt is 1-based index of top layer; convert to 0-based
    const auto jt_array = input.get_array("jt");
    const auto maxg_array = input.get_array("maxg");
    const int ktop = Kokkos::round(jt_array[0] - 1);
    const int kbot = Kokkos::round(maxg_array[0]);

    // Convection processing flags: convproc_do_aer is true; convproc_do_gas
    // can be true or false (default false unless specified)
    bool convproc_do_aer = input.get_array("convproc_do_aer")[0];
    bool convproc_do_gas = input.get_array("convproc_do_gas")[0] ;

    mam4::ConvProc::Config convproc_config;
    const int *species_class = convproc_config.species_class;
    const int *mmtoo_prevap_resusp = convproc_config.mmtoo_prevap_resusp;

    // outputs
    mam4::ColumnView dlf =
        mam4::validation::get_input_in_columnview(input, "dlf");
    mam4::wetdep::View1D aerdepwetcw("aerdepwetcw", mam4::aero_model::pcnst);
    mam4::wetdep::View1D aerdepwetis("aerdepwetis", mam4::aero_model::pcnst);
    mam4::wetdep::View1D aerdepwetis_convproc("aerdepwetis_convproc",
                                              mam4::aero_model::pcnst);
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

    // work arrays for aero_model_wetdep
    const int work_len = mam4::wetdep::get_aero_model_wetdep_work_len();
    mam4::wetdep::View1D work("work", work_len);

    // work arrays for convproc scratch1Dviews
    const int work_convproc_len = mam4::convproc::get_scratch1d_work_len(nlev);
    View1D work_convproc("work_convproc", work_convproc_len);
    Real *work_convproc_ptr = work_convproc.data();

    mam4::modal_aero_calcsize::CalcsizeData cal_data;
    cal_data.initialize();
    const bool update_mmr = true;
    cal_data.set_update_mmr(update_mmr);

    auto team_policy = mam4::ThreadTeamPolicy(1u, mam4::testing::team_size);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          Kokkos::View<Real *>
              scratch1Dviews[mam4::ConvProc::Col1DViewInd::NumScratch];
          mam4::convproc::init_scratch_1d_views(scratch1Dviews,
                                                work_convproc_ptr, nlev);

          auto progs_in = progs;
          auto tends_in = tends;

          // Inject validation values to progs
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, nlev), [&](int kk) {
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
              aerdepwetis, aerdepwetcw, work, isprx, aerdepwetis_convproc,
              // Convection mass flux parameters
              scratch1Dviews, mu_icol, md_icol, du_icol, eu_icol, ed_icol,
              dp_icol, p_del_dry_icol, ktop, kbot, convproc_do_aer,
              convproc_do_gas, species_class, mmtoo_prevap_resusp, aero_config);

          team.team_barrier();
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(team, 0, nlev), [&](int kk) {
                const auto ptend_q_kk = ekat::subview(ptend_q, kk);
                mam4::utils::extract_ptend_from_tendencies(tends_in, ptend_q_kk,
                                                           kk);
                const auto qqcw_kk = ekat::subview(qqcw, kk);
                mam4::utils::extract_qqcw_from_prognostics(progs_in, qqcw_kk,
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

    // ptend_lq in Fortran is logical flag array of length pcnst
    std::vector<Real> ptend_lq_out(mam4::aero_model::pcnst, 1.0);
    output.set("ptend_lq", ptend_lq_out);

    // Note: Fortran validation code uses -9999.9 for ptend that are not
    // aerosols when convproc_do_gas is false, but if convproc_do_gas is true,
    // trace gas species are also updated.
    using range_type = Kokkos::pair<int, int>;
    if (!convproc_do_gas) {
      const auto &ptend_q_non = Kokkos::subview(
          ptend_q, Kokkos::ALL, range_type(0, mam4::utils::aero_start_ind()));
      Kokkos::deep_copy(ptend_q_non, -9999.900390625);
    }
    std::vector<Real> output_ptend(nlev * mam4::aero_model::pcnst, 0);
    mam4::validation::convert_2d_view_device_to_1d_vector(ptend_q,
                                                          output_ptend);
    output.set("ptend_q", output_ptend);

    // Extract qqcw from prognostics into output vector
    std::vector<Real> qqcw_out(nlev * mam4::aero_model::pcnst, 0);
    auto qqcw_out_host = Kokkos::create_mirror_view(qqcw);
    Kokkos::deep_copy(qqcw_out_host, qqcw);
    int qqcw_cnt = 0;
    for (int kk = 0; kk < nlev; ++kk) {
      for (int i = 0; i < mam4::aero_model::pcnst; ++i) {
        if (i < mam4::utils::aero_start_ind()) {
          qqcw_out[qqcw_cnt] = -9999.900390625;
        } else {
          qqcw_out[qqcw_cnt] = qqcw_out_host(kk, i);
        }
        qqcw_cnt++;
      }
    }
    output.set("qqcw", qqcw_out);
  });
}
