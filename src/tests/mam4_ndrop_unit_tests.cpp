#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat_comm.hpp>
#include <ekat_logger.hpp>
#include <ekat_pack_kokkos.hpp>

using mam4::Real;

// NOTE: other than the nonnegative-ish requirements, this test is basically
// vacuous, since it mostly does the same thing as the function, but I suppose
// it'll let us know if the function changes while not having to worry about
// new/globally-defined constants
TEST_CASE("test_ndrop_init", "mam4_ndrop_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop unit tests",
                                ekat::logger::LogLevel::debug, comm);

  Real a_exp45logsig;
  Real a_alogsig;
  Real a_aten;
  Real a_num2vol_ratio_min_nmodes;
  Real a_num2vol_ratio_max_nmodes;

  Real exp45logsig[mam4::AeroConfig::num_modes()];
  Real alogsig[mam4::AeroConfig::num_modes()];
  Real aten;
  Real num2vol_ratio_min_nmodes[mam4::AeroConfig::num_modes()];
  Real num2vol_ratio_max_nmodes[mam4::AeroConfig::num_modes()];

  const Real one = 1;
  const Real two = 2;
  const Real one_thousand = 1e3;

  // close to largest double value--for determining if values are positive
  const Real huge = 1.0e308;

  mam4::ndrop::ndrop_init(exp45logsig, alogsig, aten, num2vol_ratio_min_nmodes,
                          num2vol_ratio_max_nmodes);

  for (int i = 0; i < mam4::AeroConfig::num_modes(); ++i) {
    a_alogsig = mam4::log(mam4::modes(i).mean_std_dev);
    a_exp45logsig = mam4::exp(4.5 * alogsig[i] * alogsig[i]);
    a_num2vol_ratio_min_nmodes =
        one / mam4::conversions::mean_particle_volume_from_diameter(
                  mam4::modes(i).max_diameter, mam4::modes(i).mean_std_dev);
    a_num2vol_ratio_max_nmodes =
        one / mam4::conversions::mean_particle_volume_from_diameter(
                  mam4::modes(i).min_diameter, mam4::modes(i).mean_std_dev);

    logger.debug("reference and computed alogsig[i] = {}, {}", a_alogsig,
                 alogsig[i]);
    REQUIRE(mam4::FloatingPoint<Real>::equiv(alogsig[i], a_alogsig));
    // it's a log, so 0 < x < inf \approx huge
    REQUIRE(mam4::FloatingPoint<Real>::in_bounds(alogsig[i], 0.0, huge));

    logger.debug("reference and computed exp45logsig[i] = {}, {}",
                 a_exp45logsig, exp45logsig[i]);
    REQUIRE(mam4::FloatingPoint<Real>::equiv(exp45logsig[i], a_exp45logsig));
    // has form exp(4.5 x^2) => 1 <= x < inf \approx huge
    REQUIRE(mam4::FloatingPoint<Real>::in_bounds(exp45logsig[i], 1.0, huge));

    logger.debug("reference and computed num2vol_ratio_min_nmodes[i] = {}, {}",
                 a_num2vol_ratio_min_nmodes, num2vol_ratio_min_nmodes[i]);
    REQUIRE(mam4::FloatingPoint<Real>::equiv(num2vol_ratio_min_nmodes[i],
                                             a_num2vol_ratio_min_nmodes));
    // reciprocal of a volume => should be > 0
    REQUIRE(mam4::FloatingPoint<Real>::in_bounds(num2vol_ratio_min_nmodes[i],
                                                 0.0, huge));

    logger.debug("reference and computed num2vol_ratio_max_nmodes[i] = {}, {}",
                 a_num2vol_ratio_max_nmodes, num2vol_ratio_max_nmodes[i]);
    REQUIRE(mam4::FloatingPoint<Real>::equiv(num2vol_ratio_max_nmodes[i],
                                             a_num2vol_ratio_max_nmodes));
    // reciprocal of a volume => should be > 0
    REQUIRE(mam4::FloatingPoint<Real>::in_bounds(num2vol_ratio_max_nmodes[i],
                                                 0.0, huge));
  }

  const Real rhoh2o = mam4::Constants::density_h2o;
  const Real r_universal = mam4::Constants::r_gas * one_thousand; //[J/K/kmol]
  const Real mwh2o =
      mam4::Constants::molec_weight_h2o * one_thousand; // [kg/kmol]
  a_aten = two * mwh2o * mam4::ndrop::surften /
           (r_universal * mam4::ndrop::t0 * rhoh2o);

  logger.debug("reference and computed aten = {}, {}", a_aten, aten);
  REQUIRE(mam4::FloatingPoint<Real>::equiv(aten, a_aten));
  // all quantities should be positive and nonzero (assumes surften is > 0)
  REQUIRE(mam4::FloatingPoint<Real>::in_bounds(aten, 0.0, huge));
}

TEST_CASE("test_get_aer_num", "mam4_ndrop_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop unit tests",
                                ekat::logger::LogLevel::debug, comm);

  // some of these values are made up, others are intended to be representative
  // of corresponding values in validation test inputs
  const Real voltonumbhi_amode = 4.736279937e18;
  // believe it or not, lo is indeed supposed to be larger than hi
  // it appears to be because these are calculated using dgnumlo/hi... which
  // are defined in the way that makes sense
  const Real voltonumblo_amode = 2.634717443e21;
  const int num_idx = 37;

  const Real air_density = 1.025;
  const Real vaerosol = 2.0e-13;
  const Real qcldbrn1d_num = 3.7e-13;

  // define some values to test as ans = min_max_bound(v2n_hi, v2n_lo, test_num)
  // and then work backward to get state_q
  // span the orders of magnitude with the top and bottom outside the interval
  const Real test_num[4] = {1.2e18, 3.4e19, 5.6e20, 7.9e21};
  Real state_q[mam4::aero_model::pcnst];
  Real naerosol;
  const Real ans[4] = {voltonumbhi_amode, test_num[1], test_num[2],
                       voltonumblo_amode};
  Real ans_i;

  for (int i = 0; i < 4; ++i) {
    ans_i = ans[i] * vaerosol;
    state_q[num_idx] = ((test_num[i] * vaerosol) / air_density - qcldbrn1d_num);
    mam4::ndrop::get_aer_num(voltonumbhi_amode, voltonumblo_amode, num_idx,
                             state_q, air_density, vaerosol, qcldbrn1d_num,
                             naerosol);
    logger.debug("reference value and computed naerosol = {}, {}", ans_i,
                 naerosol);
    REQUIRE(mam4::FloatingPoint<Real>::equiv(naerosol, ans_i));
  }
}

TEST_CASE("test_qsat", "mam4_ndrop_unit_tests") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  // I took these from the mam_x_validation file dropmixnuc_ts_1417.yaml
  // temperature [K]
  const Real t[mam4::AeroConfig::num_modes()] = {
      272.15, 0.2529892939E003, 0.2159618556E003, 0.2686359632E003};
  // pressure [Pa]
  const Real p[mam4::AeroConfig::num_modes()] = {
      101325.0, 0.1238254131E002, 0.1999170945E005, 0.9524671355E005};

  for (int i = 0; i < mam4::AeroConfig::num_modes(); ++i) {
    logger.debug("i = {}", i);
    logger.debug("temperature, pressure = {}, {}", t[i], p[i]);

    Real es_base = mam4::wv_sat_methods::wv_sat_svp_trans(t[i]);
    Real qs_base = mam4::wv_sat_methods::wv_sat_svp_to_qsat(es_base, p[i]);

    Real es_calc = es_base;
    Real qs_calc = qs_base;

    mam4::ndrop::qsat(t[i], p[i], es_calc, qs_calc);

    logger.debug("es [base]: [ {} ]", es_base);
    logger.debug("es [calc]: [ {} ]", es_calc);
    logger.debug("qs [base]: [ {} ]", qs_base);
    logger.debug("qs [calc]: [ {} ]", qs_calc);

    REQUIRE(
        mam4::FloatingPoint<Real>::equiv(es_calc, mam4::min(es_base, p[i])));
    REQUIRE(mam4::FloatingPoint<Real>::equiv(qs_calc, qs_base));
  }
}

TEST_CASE("test_explmix", "mam4_ndrop") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop explmix unit tests",
                                ekat::logger::LogLevel::debug, comm);

  Real q = -1;

  // set up smoke test values
  // pretend this is from one level k, in nlev, with adjacent levels k-1 (km1)
  // and k+1 (kp1)
  Real qold_km1 = 1;
  Real qold_k = 1;
  Real qold_kp1 = 1;

  Real src = 1;
  Real ek_kp1 = 1;
  Real ek_km1 = 1;
  Real overlap_kp1 = 1;
  Real overlap_km1 = 1;

  Real dt = 0.1;

  Real qactold_km1 = 1;
  Real qactold_kp1 = 1;

  q = mam4::ndrop::explmix(qold_km1, qold_k, qold_kp1, src, ek_kp1, ek_km1,
                           overlap_kp1, overlap_km1, dt);

  logger.info("q = {}", q);
  REQUIRE(mam4::FloatingPoint<Real>::equiv(q, 1.1));

  q = mam4::ndrop::explmix(qold_km1, qold_k, qold_kp1, src, ek_kp1, ek_km1,
                           overlap_kp1, overlap_km1, dt, qactold_km1,
                           qactold_kp1);

  logger.info("q = {}", q);
  REQUIRE(mam4::FloatingPoint<Real>::equiv(q, 0.9));
}

TEST_CASE("test_maxsat", "mam4_ndrop") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop maxsat unit tests",
                                ekat::logger::LogLevel::debug, comm);

  logger.info("start of maxsat test");
  static constexpr int nmodes = mam4::AeroConfig::num_modes();
  logger.info("nmodes = {}", nmodes);

  Real zeta = 0;
  Real eta[nmodes];
  Real smc[nmodes];
  Real smax = 0;

  // set up smoke test values
  for (int m = 0; m < nmodes; m++) {
    eta[m] = 0;
    smc[m] = 1;
  }

  mam4::ndrop::maxsat(zeta, eta, nmodes, smc, smax);
  logger.info("smax = {}", smax);
  REQUIRE(mam4::FloatingPoint<Real>::equiv(smax, 1e-20));

  for (int m = 0; m < nmodes; m++) {
    smc[m] = 0;
  }
  smax = 0;
  mam4::ndrop::maxsat(zeta, eta, nmodes, smc, smax);
  logger.info("smax = {}", smax);
  REQUIRE(mam4::FloatingPoint<Real>::equiv(smax, 1e-10));

  // set up smoke test values
  for (int m = 0; m < nmodes; m++) {
    eta[m] = 1;
    smc[m] = 1;
  }
  smax = 0;
  const Real double_answer = 0.4698982924029698;
  const Real single_answer = 0.46989828;

  mam4::ndrop::maxsat(zeta, eta, nmodes, smc, smax);
  logger.info("smax = {}", smax);
  logger.info("double_answer = {}, single_answer = {}", double_answer,
              single_answer);
  REQUIRE((mam4::FloatingPoint<Real>::equiv(smax, double_answer) ||
           mam4::FloatingPoint<Real>::equiv(smax, single_answer)));
}
