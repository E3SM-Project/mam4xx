// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include "testing.hpp"
#include <mam4xx/mam4.hpp>
#include <mam4xx/utils.hpp>

#include <catch2/catch.hpp>
#include <ekat_comm.hpp>
#include <ekat_logger.hpp>
#include <ekat_pack_kokkos.hpp>

// if you need something from the data/ directory
// std::string data_file = MAM4_TEST_DATA_DIR;
// #include <mam4_test_config.hpp>

using mam4::Real;

TEST_CASE("test_wv_sat_svp_to_qsat", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  Real es, p;
  // FIXME: do these numbers make sense?
  es = 101325.0 + 1;
  p = 101325.0;
  Real qs = mam4::wv_sat_methods::wv_sat_svp_to_qsat(es, p);
  ss << "qs [out]: [ ";
  ss << qs << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // function requires this is bounded between 0 and 1
  REQUIRE(qs > 0.0);
  REQUIRE(qs <= 1.0);

  es -= 500;
  qs = mam4::wv_sat_methods::wv_sat_svp_to_qsat(es, p);
  ss << "qs [out]: [ ";
  ss << qs << " ";
  ss << "]";
  logger.debug(ss.str());
  // function requires this is bounded between 0 and 1
  REQUIRE(qs > 0.0);
  REQUIRE(qs <= 1.0);
}

TEST_CASE("test_wv_sat_qsat_water", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  // NOTE: qs output comes directly from wv_sat_svp_to_qsat() (above), so we
  // won't test those values here

  mam4::AeroConfig mam4_config;
  Real es, p, qs;
  Real t[4] = {273, 298, 323, 373};
  // for temperature \in [273, 373]K (~[0, 100]C) es is \in [615, 1.009e5]
  // and final comparison is min(es, p),
  // so here we essentially test GoffGratch_svp_water().
  // es is immediately calculated by svp_water() and qs by wv_sat_svp_to_qsat(),
  // so no need to initialize.
  // we make p artificially high to test es calculation
  p = 2.0e5;
  for (int i = 0; i < 4; ++i) {
    mam4::wv_sat_methods::wv_sat_qsat_water(t[i], p, es, qs);
    ss << "temperature = " << t[i];
    logger.debug(ss.str());
    ss.str("");
    ss << "es [out]: [ ";
    ss << es << " ";
    ss << "]";
    logger.debug(ss.str());
    ss.str("");
    REQUIRE(es > 200.0);
    REQUIRE(es < 1.01e5);
  }

  // make p artificially low to test min(es, p)
  p = 10;
  for (int i = 0; i < 4; ++i) {
    mam4::wv_sat_methods::wv_sat_qsat_water(t[i], p, es, qs);
    ss << "temperature = " << t[i];
    logger.debug(ss.str());
    ss.str("");
    ss << "es [out]: [ ";
    ss << es << " ";
    ss << "]";
    logger.debug(ss.str());
    // for p > es (pressure greater than boiling point of water at 1 atm) es is
    // set to p
    REQUIRE(es == p);
  }
}

TEST_CASE("test_GoffGratch_svp_ice", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  Real t[4] = {173, 198, 223, 273};
  Real es;
  // for temperature \in [173, 273]K (~[-100, 0]C) es is \in [1.0e-3, 604]
  for (int i = 0; i < 4; ++i) {
    es = mam4::wv_sat_methods::GoffGratch_svp_ice(t[i]);
    ss << "temperature = " << t[i];
    logger.debug(ss.str());
    ss.str("");
    ss << "es [out]: [ ";
    ss << es << " ";
    ss << "]";
    logger.debug(ss.str());
    ss.str("");
    REQUIRE(es > 1.0e-3);
    REQUIRE(es < 604.0);
  }
}

TEST_CASE("test_calculate_regm_nucleati", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  Real w_vlc, Na, regm;
  // FIXME: do these numbers make sense?
  w_vlc = 0.2;
  Na = 58;
  mam4::nucleate_ice::calculate_regm_nucleati(w_vlc, Na, regm);
  ss << "regm [out]: [ ";
  ss << regm << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // regm is a threshold temperature [C], so giving it a wide range of values
  // that seem reasonable for the atmosphere
  REQUIRE(regm > -200.0);
  REQUIRE(regm < 50.0);
}

TEST_CASE("test_calculate_RHw_hf", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("nucleate_ice unit tests",
                                ekat::logger::LogLevel::debug, comm);

  std::ostringstream ss;

  mam4::AeroConfig mam4_config;
  Real temperature, lnw, RHw;
  // FIXME: do these numbers make sense?
  temperature = -40.0;
  lnw = 1.0;
  mam4::nucleate_ice::calculate_RHw_hf(temperature, lnw, RHw);
  ss << "RHw [out]: [ ";
  ss << RHw << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  // these bounds come from the rough approximations in the source code
  REQUIRE(RHw > 0.9);
  REQUIRE(RHw < 1.6);

  temperature = -100.0;
  lnw = 2.0;
  mam4::nucleate_ice::calculate_RHw_hf(temperature, lnw, RHw);
  ss << "RHw [out]: [ ";
  ss << RHw << " ";
  ss << "]";
  logger.debug(ss.str());
  ss.str("");
  REQUIRE(RHw > 0.9);
  REQUIRE(RHw < 1.6);
}

TEST_CASE("test_wv_sat_svp_trans", "mam4_nucleate_ice_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("wv_sat_svp_trans unit tests",
                                ekat::logger::LogLevel::debug, comm);

  const Real epsilon = ekat::is_single_precision<Real>::value ? 0.01 : 0.0001;
  const Real tmelt = mam4::Constants::melting_pt_h2o;
  Real temperature = tmelt + 40;
  REQUIRE(mam4::abs(7373.80964886279 - mam4::wv_sat_methods::wv_sat_svp_trans(
                                           temperature)) < epsilon);
  temperature = tmelt - 10;
  REQUIRE(mam4::abs(272.7574754946415 - mam4::wv_sat_methods::wv_sat_svp_trans(
                                            temperature)) < epsilon);
  temperature = tmelt - 30;
  REQUIRE(mam4::abs(37.94098622403198 - mam4::wv_sat_methods::wv_sat_svp_trans(
                                            temperature)) < epsilon);
}
