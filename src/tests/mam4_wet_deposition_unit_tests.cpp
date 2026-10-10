// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "testing.hpp"
#include <mam4xx/mam4.hpp>

#include <ekat_comm.hpp>
#include <ekat_logger.hpp>
#include <ekat_type_traits.hpp>

#include <catch2/catch.hpp>

using mam4::Real;

Real tol = 1e-8;

TEST_CASE("test_local_precip_production", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("wet deposition local precip production test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  mam4::Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  // TODO - Pass this to subroutine instead of whole atmosphere
  const int pver = atm.num_levels();

  mam4::ColumnView pdel = mam4::testing::create_column_view(pver);
  mam4::ColumnView source_term = mam4::testing::create_column_view(pver);
  mam4::ColumnView sink_term = mam4::testing::create_column_view(pver);
  mam4::ColumnView lprec = mam4::testing::create_column_view(pver);

  // Need to use Kokkos to initialize values
  // These arrays only have a single value in them...
  // See
  // e3sm_mam4_refactor/components/eam/src/chemistry/yaml/wetdep/local_precip_production_output_ts_355.py
  Kokkos::parallel_for(
      "intialize_values_local_precip", 1, KOKKOS_LAMBDA(const int) {
        for (int i = 0; i < pver; i++) {
          pdel(i) = 0.3395589227E+04;
          source_term(i) = 0.4201774770E-07;
          sink_term(i) = 0.7626064109E-09;
          lprec(i) = 0.0;
        }
      });

  Real gravity = mam4::Constants::gravity;
  Kokkos::parallel_for(
      "test_local_precip_production", 1, KOKKOS_LAMBDA(const int) {
        Real *pdel_device = pdel.data();
        Real *source_term_device = source_term.data();
        Real *sink_term_device = sink_term.data();
        Real *lprec_device = lprec.data();
        for (int i = 0; i < nlev; i++)
          lprec_device[i] = mam4::wetdep::local_precip_production(
              pdel_device[i], source_term_device[i], sink_term_device[i],
              gravity);
      });

  auto pdel_view = Kokkos::create_mirror_view(pdel);
  Kokkos::deep_copy(pdel_view, pdel);
  auto source_term_view = Kokkos::create_mirror_view(source_term);
  Kokkos::deep_copy(source_term_view, source_term);
  auto sink_term_view = Kokkos::create_mirror_view(sink_term);
  Kokkos::deep_copy(sink_term_view, sink_term);
  auto lprec_view = Kokkos::create_mirror_view(lprec);
  Kokkos::deep_copy(lprec_view, lprec);

  for (int i = 0; i < pver; i++) {
    REQUIRE(pdel_view(i) == Approx(0.3395589227E+04));
    REQUIRE(source_term_view(i) == Approx(0.4201774770E-07));
    REQUIRE(sink_term_view(i) == Approx(0.7626064109E-09));
    REQUIRE(lprec_view(i) == Approx(0.1428546071E-04));
  }
}

TEST_CASE("test_calculate_cloudy_volume", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("wet deposition calculate cloudy volume test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  mam4::Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();

  // Input vectors
  mam4::ColumnView cld = mam4::testing::create_column_view(pver);
  mam4::ColumnView lprec = mam4::testing::create_column_view(pver);

  // Output vectors
  mam4::ColumnView cldv = mam4::testing::create_column_view(pver);
  mam4::ColumnView sumppr_all = mam4::testing::create_column_view(pver);

  // Reference input from
  // e3sm_mam4_refactor/components/eam/src/chemistry/yaml/wetdep/calculate_cloudy_volume_output_ts_355.py
  const Real cld_input[] = {
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.7444139176E-04,
      0.2265240125E-02, 0.9407597400E-02, 0.3062400200E-01, 0.3701026715E-01,
      0.9928646676E+00, 0.4534920173E+00, 0.7899418112E-01, 0.6152565646E-03,
      0.1299316699E-02, 0.4333328343E-02, 0.1001469750E-01, 0.8280038477E-02,
      0.7671868357E-02, 0.6967937341E-02, 0.6453654844E-02, 0.6271733391E-02,
      0.1966933729E+00, 0.6068935297E+00, 0.1996602881E+00, 0.5524247784E-01,
      0.1083053617E+00, 0.9356051587E+00, 0.7653835562E+00, 0.1000000000E+01,
      0.4382579369E-01, 0.4228986641E-01, 0.4131862680E-01, 0.4042474889E-01,
      0.3961324927E-01, 0.3889032712E-01, 0.3826315055E-01, 0.3773983236E-01,
      0.3731159034E-01, 0.3693081488E-01, 0.3655626956E-01, 0.3618828266E-01,
      0.3582729080E-01, 0.3547376118E-01, 0.3512820282E-01, 0.3479119174E-01,
      0.3446334629E-01, 0.3414532611E-01, 0.3383788353E-01, 0.3354181543E-01,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00};
  const Real lprec_input[] = {
      0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00,
      0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00,
      0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00,
      0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00,
      0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00,
      0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00,
      0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00,
      -0.1465471379E-22, 0.5141086008E-17,  0.2902950651E-13,
      0.9608732745E-12,  0.4332558171E-11,  0.2898600642E-10,
      0.6625175956E-10,  0.7057048577E-09,  0.5824754979E-09,
      0.2010549472E-09,  0.1062694147E-08,  0.8062030943E-08,
      0.1406566580E-07,  0.1064872777E-06,  0.4122632366E-06,
      0.1122819511E-05,  0.1375139701E-05,  0.1794852390E-05,
      0.1544126880E-05,  0.9633217442E-06,  0.1038883311E-05,
      0.1903362890E-05,  0.4421880251E-05,  0.6030320137E-05,
      0.5899572820E-05,  0.4976381681E-05,  0.1466040055E-06,
      0.2056212393E-04,  0.1663814469E-04,  0.1428546071E-04,
      0.1195696385E-04,  0.9611052184E-05,  0.7841592326E-05,
      0.6594919089E-05,  0.5267018244E-05,  0.4040073926E-05,
      0.3323013702E-05,  0.3016661397E-05,  0.2673948304E-05,
      0.2322637737E-05,  0.1963263728E-05,  0.1590232561E-05,
      0.1205314875E-05,  0.8189237560E-06,  0.4281168664E-06,
      0.3411842407E-07,  -0.3981848733E-06, -0.9384763317E-06,
      -0.7400514544E-06, -0.1409260146E-07, -0.7561162623E-08};

  // Reference solution from
  // e3sm_mam4_refactor/components/eam/src/chemistry/yaml/wetdep/calculate_cloudy_volume_output_ts_355.py
  const Real cldv_ref[] = {
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00, 0.7444139176E-04,
      0.2265240125E-02, 0.9407597400E-02, 0.3062400200E-01, 0.3701026715E-01,
      0.9928646676E+00, 0.8732264717E+00, 0.6971784484E+00, 0.6189992093E+00,
      0.3712497037E+00, 0.9288433658E-01, 0.4262124063E-01, 0.1617004013E-01,
      0.1018554355E-01, 0.8491779820E-02, 0.7802809496E-02, 0.7302114281E-02,
      0.1966933729E+00, 0.6068935297E+00, 0.1996602881E+00, 0.1210375198E+00,
      0.1083053617E+00, 0.9356051587E+00, 0.7653835562E+00, 0.1000000000E+01,
      0.3657812455E+00, 0.2392575747E+00, 0.1917354367E+00, 0.1659232671E+00,
      0.1501614368E+00, 0.1400246334E+00, 0.1329850344E+00, 0.1277466475E+00,
      0.1239394287E+00, 0.1212170584E+00, 0.1190932953E+00, 0.1172475556E+00,
      0.1156720267E+00, 0.1143463811E+00, 0.1132548099E+00, 0.1123887151E+00,
      0.1117421381E+00, 0.1113071033E+00, 0.1110806866E+00, 0.1110626278E+00,
      0.1107595912E+00, 0.1100453684E+00, 0.1094821560E+00, 0.1094714309E+00};
  const Real sumppr_all_ref[] = {
      0.0000000000E+00, 0.0000000000E+00,  0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00,  0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00,  0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00,  0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, 0.0000000000E+00,  0.0000000000E+00, 0.0000000000E+00,
      0.0000000000E+00, -0.1465471379E-22, 0.5141071354E-17, 0.2903464758E-13,
      0.9899079221E-12, 0.5322466093E-11,  0.3430847251E-10, 0.1005602321E-09,
      0.8062650898E-09, 0.1388740588E-08,  0.1589795535E-08, 0.2652489682E-08,
      0.1071452063E-07, 0.2478018642E-07,  0.1312674641E-06, 0.5435307007E-06,
      0.1666350212E-05, 0.3041489913E-05,  0.4836342302E-05, 0.6380469182E-05,
      0.7343790926E-05, 0.8382674237E-05,  0.1028603713E-04, 0.1470791738E-04,
      0.2073823752E-04, 0.2663781034E-04,  0.3161419202E-04, 0.3176079602E-04,
      0.5232291995E-04, 0.6896106464E-04,  0.8324652534E-04, 0.9520348919E-04,
      0.1048145414E-03, 0.1126561337E-03,  0.1192510528E-03, 0.1245180710E-03,
      0.1285581450E-03, 0.1318811587E-03,  0.1348978201E-03, 0.1375717684E-03,
      0.1398944061E-03, 0.1418576698E-03,  0.1434479024E-03, 0.1446532173E-03,
      0.1454721410E-03, 0.1459002579E-03,  0.1459343763E-03, 0.1455361914E-03,
      0.1445977151E-03, 0.1438576637E-03,  0.1438435710E-03, 0.1438360099E-03};

  // Need to use Kokkos to initialize values
  Kokkos::parallel_for(
      "intialize_values_local_precip", 1, KOKKOS_LAMBDA(const int) {
        for (int i = 0; i < pver; i++) {
          cld(i) = cld_input[i];
          lprec(i) = lprec_input[i];
        }
      });

  Kokkos::parallel_for(
      "test_calculate_cloudy_volume_true", 1, KOKKOS_LAMBDA(const int) {
        Real *lprec_device = lprec.data();
        Real *sumppr_all_device = sumppr_all.data();
        // True is the only flag with validation data available
        auto lprec = [&](int i) { return lprec_device[i]; };
        mam4::wetdep::calculate_cloudy_volume(nlev, cld, lprec, true, cldv);
        sumppr_all_device[0] = lprec_device[0];
        for (int i = 1; i < nlev; i++)
          sumppr_all_device[i] = sumppr_all_device[i - 1] + lprec_device[i];
      });

  auto cld_view = Kokkos::create_mirror_view(cld);
  Kokkos::deep_copy(cld_view, cld);
  auto lprec_view = Kokkos::create_mirror_view(lprec);
  Kokkos::deep_copy(lprec_view, lprec);
  auto cldv_view = Kokkos::create_mirror_view(cldv);
  Kokkos::deep_copy(cldv_view, cldv);
  auto sumppr_all_view = Kokkos::create_mirror_view(sumppr_all);
  Kokkos::deep_copy(sumppr_all_view, sumppr_all);

  for (int i = 0; i < pver; i++) {
    REQUIRE(cld_view(i) == Approx(cld_input[i]));
    REQUIRE(lprec_view(i) == Approx(lprec_input[i]));
    REQUIRE(cldv_view(i) == Approx(cldv_ref[i]));
    const double scale = std::abs(sumppr_all_ref[i]);
    REQUIRE(sumppr_all_view(i) == Approx(sumppr_all_ref[i]).scale(scale));
  }
}

TEST_CASE("test_rain_mix_ratio", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("rain mixing ratio test",
                                ekat::logger::LogLevel::debug, comm);
  int nlev = 72;
  Real pblh = 1000;
  mam4::Atmosphere atm = mam4::testing::create_atmosphere(nlev, pblh);

  const int pver = atm.num_levels();

  // Input Vectors
  mam4::ColumnView temperature = mam4::testing::create_column_view(pver);
  mam4::ColumnView pmid = mam4::testing::create_column_view(pver);
  mam4::ColumnView sumppr = mam4::testing::create_column_view(pver);

  // Output Vectors
  mam4::ColumnView rain = mam4::testing::create_column_view(pver);

  // Need to use Kokkos to initialize values
  // Validation data from
  // e3sm_mam4_refactor/components/eam/src/chemistry/yaml/wetdep/rain_mix_ratio_output_ts_355.py
  Kokkos::parallel_for(
      "intialize_values_local_precip", 1, KOKKOS_LAMBDA(const int) {
        for (int i = 0; i < pver; i++) {
          temperature(i) = 0.2804261386E+03;
          pmid(i) = 0.6753476429E+05;
          sumppr(i) = 0.8324652534E-04;
        }
      });

  Kokkos::parallel_for(
      "rain_mix_ratio_test", 1, KOKKOS_LAMBDA(const int) {
        Real *temperature_device = temperature.data();
        Real *pmid_device = pmid.data();
        Real *sumppr_device = sumppr.data();
        Real *rain_device = rain.data();
        for (int i = 0; i < nlev; ++i)
          rain_device[i] = mam4::wetdep::rain_mix_ratio(
              temperature_device[i], pmid_device[i], sumppr_device[i]);
      });

  auto temperature_view = Kokkos::create_mirror_view(temperature);
  Kokkos::deep_copy(temperature_view, temperature);
  auto pmid_view = Kokkos::create_mirror_view(pmid);
  Kokkos::deep_copy(pmid_view, pmid);
  auto sumppr_view = Kokkos::create_mirror_view(sumppr);
  Kokkos::deep_copy(sumppr_view, sumppr);
  auto rain_view = Kokkos::create_mirror_view(rain);
  Kokkos::deep_copy(rain_view, rain);

  for (int i = 0; i < pver; i++) {
    REQUIRE(temperature_view(i) == Approx(0.2804261386E+03));
    REQUIRE(pmid_view(i) == Approx(0.6753476429E+05));
    REQUIRE(sumppr_view(i) == Approx(0.8324652534E-04));
    REQUIRE(rain_view(i) == Approx(0.1351673886E-04));
  }
}
TEST_CASE("test_flux_precnum_vs_flux_prec_mpln(",
          "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "wet deposition flux_precnum_vs_flux_prec_mpln( test",
      ekat::logger::LogLevel::debug, comm);
  Kokkos::parallel_for(
      "flux_precnum_vs_flux_prec_mpln(", 1, KOKKOS_LAMBDA(const int) {
        {
          const Real flux_prec = 0.2804261386e+03;
          const int  jstrcnv = 1;
          const Real flux = 625774.9256400075;
	  const Real ans = mam4::wetdep::flux_precnum_vs_flux_prec_mpln(flux_prec,jstrcnv);
	  const Real err = mam4::abs((flux-ans)/ans);
          EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real flux_prec = 0.2804261386e+03;
  const int jstrcnv = 2;
  const Real flux = 222562.1254970778;
  const Real ans =
      mam4::wetdep::flux_precnum_vs_flux_prec_mpln(flux_prec, jstrcnv);
  const Real err = mam4::abs((flux - ans) / ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real flux_prec = 1.e-37;
  const int jstrcnv = 1;
  const Real flux = 0.0;
  const Real ans =
      mam4::wetdep::flux_precnum_vs_flux_prec_mpln(flux_prec, jstrcnv);
  const Real err = mam4::abs(flux - ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real flux_prec = 1.e-37;
  const int jstrcnv = 2;
  const Real flux = 0.0;
  const Real ans =
      mam4::wetdep::flux_precnum_vs_flux_prec_mpln(flux_prec, jstrcnv);
  const Real err = mam4::abs(flux - ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
});
}
TEST_CASE("faer_resusp_vs_fprec_evap_mpln(", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "wet deposition faer_resusp_vs_fprec_evap_mpln test",
      ekat::logger::LogLevel::debug, comm);
  Kokkos::parallel_for(
      "faer_resusp_vs_fprec_evap_mpln", 1, KOKKOS_LAMBDA(const int) {
        {
          const Real fprec_evap = 0.1;
          const int  jstrcnv = 1;
          const Real flux = 0.007075389488885791;
	  const Real ans = mam4::wetdep::faer_resusp_vs_fprec_evap_mpln(fprec_evap,jstrcnv);
	  const Real err = mam4::abs((flux-ans)/ans);
          EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real fprec_evap = 0.2;
  const int jstrcnv = 2;
  const Real flux = 0.009535933995416133;
  const Real ans =
      mam4::wetdep::faer_resusp_vs_fprec_evap_mpln(fprec_evap, jstrcnv);
  const Real err = mam4::abs((flux - ans) / ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real fprec_evap = 1.0e-2;
  const int jstrcnv = 1;
  const Real flux = 0.0005124494240644202;
  const Real ans =
      mam4::wetdep::faer_resusp_vs_fprec_evap_mpln(fprec_evap, jstrcnv);
  const Real err = mam4::abs((flux - ans) / ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real fprec_evap = 1.0e-2;
  const int jstrcnv = 2;
  const Real flux = 6.222788982804435e-05;
  const Real ans =
      mam4::wetdep::faer_resusp_vs_fprec_evap_mpln(fprec_evap, jstrcnv);
  const Real err = mam4::abs((flux - ans) / ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
});
}
TEST_CASE("fprecn_resusp_vs_fprec_evap_mpln(", "mam4_wet_deposition_process") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger(
      "wet deposition fprecn_resusp_vs_fprec_evap_mpln test",
      ekat::logger::LogLevel::debug, comm);
  Kokkos::parallel_for(
      "fprecn_resusp_vs_fprec_evap_mpln", 1, KOKKOS_LAMBDA(const int) {
        {
          const Real fprec_evap = 0.1;
          const int  jstrcnv = 1;
          const Real flux = 0.2768051337282046;
	  const Real ans = mam4::wetdep::fprecn_resusp_vs_fprec_evap_mpln(fprec_evap,jstrcnv);
	  const Real err = mam4::abs((flux-ans)/ans);
          EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real fprec_evap = 0.2;
  const int jstrcnv = 2;
  const Real flux = 0.1183317577400569;
  const Real ans =
      mam4::wetdep::fprecn_resusp_vs_fprec_evap_mpln(fprec_evap, jstrcnv);
  const Real err = mam4::abs((flux - ans) / ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real fprec_evap = 1.0e-2;
  const int jstrcnv = 1;
  const Real flux = 0.03401171698136975;
  const Real ans =
      mam4::wetdep::fprecn_resusp_vs_fprec_evap_mpln(fprec_evap, jstrcnv);
  const Real err = mam4::abs((flux - ans) / ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
{
  const Real fprec_evap = 1.0e-2;
  const int jstrcnv = 2;
  const Real flux = 0.002724799476656648;
  const Real ans =
      mam4::wetdep::fprecn_resusp_vs_fprec_evap_mpln(fprec_evap, jstrcnv);
  const Real err = mam4::abs((flux - ans) / ans);
  EKAT_KERNEL_REQUIRE(err < 1.0e-6);
}
});
}
