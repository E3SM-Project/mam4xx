#include <mam4xx/aero_model_emissions.hpp>
#include <mam4xx/floating_point.hpp>
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat_comm.hpp>
#include <ekat_logger.hpp>

#include <limits>

using mam4::Real;

TEST_CASE("test_init_dust_dmt_vwr", "mam4_aero_emissions_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "aero_model_emissions unit tests: test_init_dust_dmt_vwr",
      ekat::logger::LogLevel::debug, comm);

  const int dust_nbin = mam4::aero_model_emissions::dust_nbin;
  // const Real dust_dmt_grd[dust_nbin + 1] = {1.0e-7, 1.0e-6, 1.0e-5};
  //  Real dust_dmt_vwr[dust_nbin];
  Real dust_dmt_vwr_ref[dust_nbin] = {0.78056703442146215e-6,
                                      0.38983341139985417e-5};

  mam4::aero_model_emissions::DustEmissionsData dust_data;
  // aero_model_emissions::init_dust_dmt_vwr(dust_dmt_grd, dust_dmt_vwr);
  mam4::aero_model_emissions::init_dust_dmt_vwr(dust_data);

  for (int i = 0; i < dust_nbin; ++i) {
    logger.debug("computed value of dust_dmt_vwr[{}] = {}", i,
                 dust_data.dust_dmt_vwr[i]);
    logger.debug("reference value of dust_dmt_vwr[{}] = {}", i,
                 dust_dmt_vwr_ref[i]);
  }

  for (int i = 0; i < dust_nbin; ++i) {
    REQUIRE(mam4::FloatingPoint<Real>::equiv(dust_data.dust_dmt_vwr[i],
                                             dust_dmt_vwr_ref[i]));
  }
}

TEST_CASE("test_calc_org_matter_seasalt_competitive_adsorption",
          "mam4_aero_emissions_unit_tests") {
  using namespace mam4::aero_model_emissions;

  constexpr Real sample_mpoly = 29.462131363967192;
  constexpr Real sample_mprot = 8.838639409190158;
  constexpr Real sample_mlip = 0.016230117959603003;
  constexpr Real expected_class_fraction[n_organic_species] = {
      0.017965715405036733, 0.0857795033139621, 0.018650121485877733};
  constexpr Real expected_total_fraction = 0.12239534020487657;
  const Real tolerance = 128 * std::numeric_limits<Real>::epsilon();

  SeasaltEmissionsData data;
  init_seasalt(data);
  data.mpoly = sample_mpoly;
  data.mprot = sample_mprot;
  data.mlip = sample_mlip;

  Real mass_frac_bub_section[n_organic_species_max][salt_nsection] = {{0.0}};
  Real om_seasalt[salt_nsection] = {0.0};
  calc_org_matter_seasalt(data, mass_frac_bub_section, om_seasalt);

  for (int ibin = 0; ibin < salt_nsection; ++ibin) {
    const bool selected = data.Dg(ibin) >= data.seasalt_size_range_lo(1) &&
                          data.Dg(ibin) < data.seasalt_size_range_hi(0);
    const Real expected_total = selected ? expected_total_fraction : 0.0;
    REQUIRE(mam4::FloatingPoint<Real>::rel(om_seasalt[ibin], expected_total,
                                           tolerance));
    for (int iorg = 0; iorg < n_organic_species; ++iorg) {
      const Real expected = selected ? expected_class_fraction[iorg] : 0.0;
      REQUIRE(mam4::FloatingPoint<Real>::rel(mass_frac_bub_section[iorg][ibin],
                                             expected, tolerance));
    }
  }
}

TEST_CASE("test_calc_org_matter_seasalt_coverage_is_bounded",
          "mam4_aero_emissions_unit_tests") {
  using namespace mam4::aero_model_emissions;

  SeasaltEmissionsData data;
  init_seasalt(data);
  data.mpoly = 10.0 * 29.462131363967192;
  data.mprot = 10.0 * 8.838639409190158;
  data.mlip = 10.0 * 0.016230117959603003;

  Real mass_frac_bub_section[n_organic_species_max][salt_nsection] = {{0.0}};
  Real om_seasalt[salt_nsection] = {0.0};
  calc_org_matter_seasalt(data, mass_frac_bub_section, om_seasalt);

  // Recover theta from the uncapped class fractions in one selected section.
  // This checks the Eq. 2 coverage invariant without exposing a new production
  // interface solely for testing.
  constexpr int selected_bin = 0;
  const Real salt_surface_mass = vol_density_NaCl_seawater * l_bub;
  const Real fraction_denominator =
      salt_surface_mass / (1.0 - om_seasalt[selected_bin]);
  Real theta_sum = 0.0;
  for (int iorg = 0; iorg < n_organic_species; ++iorg) {
    theta_sum += mass_frac_bub_section[iorg][selected_bin] *
                 fraction_denominator / (2.0 * data.dens_srf_org(iorg));
  }

  const Real tolerance = 128 * std::numeric_limits<Real>::epsilon();
  REQUIRE(
      mam4::FloatingPoint<Real>::rel(theta_sum, 0.5122028187397498, tolerance));
  REQUIRE(theta_sum >= 0.0);
  REQUIRE(theta_sum < 1.0);
}

TEST_CASE("test_calc_org_matter_seasalt_zero_and_cap",
          "mam4_aero_emissions_unit_tests") {
  using namespace mam4::aero_model_emissions;

  SeasaltEmissionsData data;
  init_seasalt(data);
  data.mpoly = 0.0;
  data.mprot = 0.0;
  data.mlip = 0.0;

  Real mass_frac_bub_section[n_organic_species_max][salt_nsection] = {{0.0}};
  Real om_seasalt[salt_nsection] = {0.0};
  calc_org_matter_seasalt(data, mass_frac_bub_section, om_seasalt);
  for (int ibin = 0; ibin < salt_nsection; ++ibin) {
    REQUIRE(om_seasalt[ibin] == 0.0);
    for (int iorg = 0; iorg < n_organic_species; ++iorg) {
      REQUIRE(mass_frac_bub_section[iorg][ibin] == 0.0);
    }
  }

  data.mpoly = 10000.0;
  calc_org_matter_seasalt(data, mass_frac_bub_section, om_seasalt);
  constexpr int selected_bin = 0;
  const Real tolerance = 128 * std::numeric_limits<Real>::epsilon();
  REQUIRE(mam4::FloatingPoint<Real>::rel(om_seasalt[selected_bin], 0.78,
                                         tolerance));
  REQUIRE(mam4::FloatingPoint<Real>::rel(mass_frac_bub_section[0][selected_bin],
                                         0.78, tolerance));
  REQUIRE(mass_frac_bub_section[1][selected_bin] == 0.0);
  REQUIRE(mass_frac_bub_section[2][selected_bin] == 0.0);
}

TEST_CASE("test_marine_organic_emissions_mode_mapping",
          "mam4_aero_emissions_unit_tests") {
  using namespace mam4::aero_model_emissions;

  constexpr Real expected_total_fraction = 0.12239534020487657;
  constexpr Real ocean_fraction = 0.4;
  const Real organic_to_salt =
      expected_total_fraction / (1.0 - expected_total_fraction);
  const Real tolerance = 128 * std::numeric_limits<Real>::epsilon();

  SeasaltEmissionsData data;
  init_seasalt(data);
  data.mpoly = 29.462131363967192;
  data.mprot = 8.838639409190158;
  data.mlip = 0.016230117959603003;
  data.seasalt_emis_scale_factor = 0.6;

  Real fi[salt_nsection];
  for (int ibin = 0; ibin < salt_nsection; ++ibin) {
    fi[ibin] = 1.0;
  }
  const bool emit_this_mode[organic_num_modes] = {true, true, false};
  Real cflux[pcnst] = {0.0};
  marine_organic_emissions(fi, ocean_fraction, data, emit_this_mode, cflux);

  for (int ispec = 0; ispec < 2; ++ispec) {
    Real expected_number = 0.0;
    Real expected_mass = 0.0;
    const int range_index = nsalt + ispec;
    for (int ibin = 0; ibin < salt_nsection; ++ibin) {
      if (data.Dg(ibin) >= data.seasalt_size_range_lo(range_index) &&
          data.Dg(ibin) < data.seasalt_size_range_hi(range_index)) {
        const Real number_flux =
            fi[ibin] * ocean_fraction * data.seasalt_emis_scale_factor;
        expected_number += number_flux * organic_to_salt;
        expected_mass += number_flux * (4.0 / 3.0) * mam4::Constants::pi *
                         mam4::pow(data.rdry[ibin], 3) * seasalt_density *
                         organic_to_salt;
      }
    }

    const int mass_index = data.seasalt_indices(nsalt + ispec);
    const int number_index =
        data.seasalt_indices(nsalt + nsalt_om + data.organic_num_idx(ispec));
    REQUIRE(mam4::FloatingPoint<Real>::rel(cflux[mass_index], expected_mass,
                                           tolerance));
    REQUIRE(mam4::FloatingPoint<Real>::rel(cflux[number_index], expected_number,
                                           tolerance));
  }

  const int disabled_mass_index = data.seasalt_indices(nsalt + 2);
  const int disabled_number_index =
      data.seasalt_indices(nsalt + nsalt_om + data.organic_num_idx(2));
  REQUIRE(cflux[disabled_mass_index] == 0.0);
  REQUIRE(cflux[disabled_number_index] == 0.0);
}
