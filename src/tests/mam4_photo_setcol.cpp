// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "atmosphere_utils.hpp"
#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat_logger.hpp>
#include <random>

// Precision-dependent tolerance traits
template <typename T> struct PrecisionTolerance;

template <> struct PrecisionTolerance<float> {
  static constexpr float tol = 1e-5f; // Single precision tolerance
};

template <> struct PrecisionTolerance<double> {
  static constexpr double tol = 1e-12; // Double precision tolerance
};

using mam4::Real;

#ifdef MAM4XX_ENABLE_GPU
constexpr int team_size = mam4::nlev;
#else
constexpr int team_size = 1;
#endif

// Test compute_o3_column_density: serial reference vs parallel implementation
TEST_CASE("compute_o3_column_density", "mo_photo") {

  ekat::Comm comm;

  ekat::logger::Logger<> logger("compute_o3_column_density tests",
                                ekat::logger::LogLevel::debug, comm);
  using View1D = mam4::DeviceType::view_1d<Real>;
  using View1DHost = typename mam4::HostType::view_1d<Real>;

  // Initialize random number generator for reproducibility
  std::default_random_engine generator(98765);
  std::uniform_real_distribution<double> unif_dist(1e-8, 1e-5);

  constexpr int pver = mam4::nlev;
  constexpr Real tol = PrecisionTolerance<Real>::tol;

  // Test inputs
  const Real o3_col_deltas_0 =
      1.6e15;                    // Top-of-atmosphere column density [1/cm²]
  const Real mw_o3 = 48.0 / 1e3; // Ozone molecular weight [kg/mol]
  constexpr Real xfactor = 2.8704e21 / (9.80616 * 1.38044); // From set_ub_col

  // Atmospheric profile parameters (humid atmosphere with RH ~32%-98%)
  Real pblh = 1000;         // Planetary boundary layer height [m]
  const Real Tv0 = 300;     // Reference virtual temperature [K]
  const Real Gammav = 0.01; // Virtual temperature lapse rate [K/m]
  const Real qv0 = 0.015;   // Surface specific humidity [kg/kg]
  const Real qv1 = 7.5e-4;  // Specific humidity lapse rate [1/m]
  mam4::Atmosphere atm =
      mam4::init_atm_const_tv_lapse_rate(pver, pblh, Tv0, Gammav, qv0, qv1);

  // Create host views for inputs
  View1DHost pdel_host("pdel_host", pver);     // Pressure thickness [Pa]
  View1DHost mmr_o3_host("mmr_o3_host", pver); // Mass mixing ratio [kg/kg]
  View1DHost o3_col_dens_ref("o3_col_dens_ref", pver); // Reference output

  const auto pdel = atm.hydrostatic_dp; // Pressure thickness array [Pa]
  Kokkos::deep_copy(pdel_host, pdel);
  // Initialize random input data
  for (int k = 0; k < pver; ++k) {
    mmr_o3_host(k) = unif_dist(generator);
  }

  // ===== SERIAL REFERENCE IMPLEMENTATION =====
  // Step 1: Compute column density deltas for each level
  View1DHost o3_col_deltas_host("o3_col_deltas_host", pver + 1);
  o3_col_deltas_host(0) = o3_col_deltas_0;

  Real running_sum = 0.0;
  for (int kk = 0; kk < mam4::nlev; ++kk) {
    const Real vmr_o3_kk =
        mam4::conversions::vmr_from_mmr(mmr_o3_host(kk), mw_o3);
    const Real delta_kk = xfactor * pdel_host(kk) * vmr_o3_kk;

    o3_col_dens_ref(kk) = o3_col_deltas_0 + running_sum + 0.5 * delta_kk;

    running_sum += delta_kk;
  }

  // ===== PARALLEL IMPLEMENTATION USING compute_o3_column_density =====
  // Copy inputs to device
  View1D mmr_o3("mmr_o3", pver);
  View1D o3_col_dens("o3_col_dens", pver);
  Kokkos::deep_copy(mmr_o3, mmr_o3_host);

  auto team_policy = mam4::ThreadTeamPolicy(1, team_size);
  Kokkos::parallel_for(
      "compute_o3_column", team_policy,
      KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
        mam4::microphysics::compute_o3_column_density(
            team,
            pdel,            // Pressure thickness array [nlev]
            mmr_o3,          // Ozone mass mixing ratio [nlev]
            o3_col_deltas_0, // Top-of-atmosphere column density [1/cm²]
            mw_o3,           // Ozone molecular weight [kg/mol]
            o3_col_dens      // Output: column density [1/cm²] [nlev]
        );
      });
  Kokkos::fence();

  // Copy results back to host for comparison
  auto o3_col_dens_host = Kokkos::create_mirror_view(o3_col_dens);
  Kokkos::deep_copy(o3_col_dens_host, o3_col_dens);

  // Compare serial reference with parallel implementation

  for (int k = 0; k < pver; ++k) {
    const Real raw_diff = o3_col_dens_host(k) - o3_col_dens_ref(k);
    const Real diff =
        (raw_diff > 0 ? raw_diff : -raw_diff) / o3_col_dens_ref(k);
    if (diff >= tol) {
      std::ostringstream ss;
      ss << "diff : [ ";
      ss << "Level " << k << ": diff = " << diff
         << ", parallel = " << o3_col_dens_host(k)
         << ", serial = " << o3_col_dens_ref(k) << "\n";
      ss << "]";
      logger.debug(ss.str());
    }
    REQUIRE(diff <= tol);
  }
}

// ============================================================================
// Helpers shared by interpolate_rsf and jlong tests
// ============================================================================

// using namespace mam4;

// Table dimensions — compile-time constants required inside device lambdas.
constexpr int test_nw = 4;
constexpr int test_numj = 1;
constexpr int test_nump = 5;
constexpr int test_numsza = 3;
constexpr int test_numcolo3 = 4;
constexpr int test_numalb = 3;
constexpr int test_nt = 201;
constexpr int test_np_xs = 4;
constexpr Real test_sza_in = 30.0; // degrees, well within daylight range

// Build a small PhotoTableData with synthetic monotone lookup arrays.
static mam4::mo_photo::PhotoTableData build_test_photo_table() {

  auto photo_table = mam4::mo_photo::create_photo_table_data(
      test_nw, test_nt, test_np_xs, test_numj, test_nump, test_numsza,
      test_numcolo3, test_numalb);

  std::default_random_engine gen(54321);
  std::uniform_real_distribution<Real> pos(1e-6, 1.0);

  // sza [degrees]: monotone increasing
  auto sza_h = Kokkos::create_mirror_view(photo_table.sza);
  sza_h(0) = 0.0;
  sza_h(1) = 45.0;
  sza_h(2) = 88.0;
  Kokkos::deep_copy(photo_table.sza, sza_h);

  auto del_sza_h = Kokkos::create_mirror_view(photo_table.del_sza);
  for (int i = 0; i < test_numsza - 1; ++i)
    del_sza_h(i) = 1.0 / (sza_h(i + 1) - sza_h(i));
  Kokkos::deep_copy(photo_table.del_sza, del_sza_h);

  // alb: monotone increasing
  auto alb_h = Kokkos::create_mirror_view(photo_table.alb);
  alb_h(0) = 0.0;
  alb_h(1) = 0.3;
  alb_h(2) = 0.8;
  Kokkos::deep_copy(photo_table.alb, alb_h);

  auto del_alb_h = Kokkos::create_mirror_view(photo_table.del_alb);
  for (int i = 0; i < test_numalb - 1; ++i)
    del_alb_h(i) = 1.0 / (alb_h(i + 1) - alb_h(i));
  Kokkos::deep_copy(photo_table.del_alb, del_alb_h);

  // press [hPa]: monotone decreasing (surface first)
  auto press_h = Kokkos::create_mirror_view(photo_table.press);
  press_h(0) = 1000.0;
  press_h(1) = 500.0;
  press_h(2) = 100.0;
  press_h(3) = 50.0;
  press_h(4) = 10.0;
  Kokkos::deep_copy(photo_table.press, press_h);

  auto del_p_h = Kokkos::create_mirror_view(photo_table.del_p);
  for (int i = 0; i < test_nump - 1; ++i)
    del_p_h(i) = 1.0 / (press_h(i) - press_h(i + 1));
  Kokkos::deep_copy(photo_table.del_p, del_p_h);

  // colo3: decreasing with altitude
  auto colo3_h = Kokkos::create_mirror_view(photo_table.colo3);
  colo3_h(0) = 5e17;
  colo3_h(1) = 4e17;
  colo3_h(2) = 3e17;
  colo3_h(3) = 2e17;
  colo3_h(4) = 1e17;
  Kokkos::deep_copy(photo_table.colo3, colo3_h);

  // o3rat: monotone increasing
  auto o3rat_h = Kokkos::create_mirror_view(photo_table.o3rat);
  o3rat_h(0) = 0.5;
  o3rat_h(1) = 1.0;
  o3rat_h(2) = 1.5;
  o3rat_h(3) = 2.0;
  Kokkos::deep_copy(photo_table.o3rat, o3rat_h);

  auto del_o3rat_h = Kokkos::create_mirror_view(photo_table.del_o3rat);
  for (int i = 0; i < test_numcolo3 - 1; ++i)
    del_o3rat_h(i) = 1.0 / (o3rat_h(i + 1) - o3rat_h(i));
  Kokkos::deep_copy(photo_table.del_o3rat, del_o3rat_h);

  // etfphot: positive random, size nw
  auto etfphot_h = Kokkos::create_mirror_view(photo_table.etfphot);
  for (int w = 0; w < test_nw; ++w)
    etfphot_h(w) = 1e13 * (0.8 + pos(gen));
  Kokkos::deep_copy(photo_table.etfphot, etfphot_h);

  // rsf_tab(nw, nump, numsza, numcolo3, numalb): random positive
  auto rsf_tab_h = Kokkos::create_mirror_view(photo_table.rsf_tab);
  for (int w = 0; w < test_nw; ++w)
    for (int ip = 0; ip < test_nump; ++ip)
      for (int is = 0; is < test_numsza; ++is)
        for (int iv = 0; iv < test_numcolo3; ++iv)
          for (int ia = 0; ia < test_numalb; ++ia)
            rsf_tab_h(w, ip, is, iv, ia) = pos(gen);
  Kokkos::deep_copy(photo_table.rsf_tab, rsf_tab_h);

  // prs [hPa]: monotone decreasing
  auto prs_h = Kokkos::create_mirror_view(photo_table.prs);
  prs_h(0) = 500.0;
  prs_h(1) = 100.0;
  prs_h(2) = 10.0;
  prs_h(3) = 1.0;
  Kokkos::deep_copy(photo_table.prs, prs_h);

  auto dprs_h = Kokkos::create_mirror_view(photo_table.dprs);
  for (int i = 0; i < test_np_xs - 1; ++i)
    dprs_h(i) = 1.0 / (prs_h(i) - prs_h(i + 1));
  Kokkos::deep_copy(photo_table.dprs, dprs_h);

  auto pam_h = Kokkos::create_mirror_view(photo_table.pht_alias_mult_1);
  pam_h(0) = 1.0;
  Kokkos::deep_copy(photo_table.pht_alias_mult_1, pam_h);

  auto li_h = Kokkos::create_mirror_view(photo_table.lng_indexer);
  li_h(0) = 0;
  Kokkos::deep_copy(photo_table.lng_indexer, li_h);

  // xsqy(numj, nw, nt, np_xs): random positive
  auto xsqy_h = Kokkos::create_mirror_view(photo_table.xsqy);
  for (int j = 0; j < test_numj; ++j)
    for (int w = 0; w < test_nw; ++w)
      for (int it = 0; it < test_nt; ++it)
        for (int ip = 0; ip < test_np_xs; ++ip)
          xsqy_h(j, w, it, ip) = pos(gen);
  Kokkos::deep_copy(photo_table.xsqy, xsqy_h);

  return photo_table;
}

// ============================================================================
// Test: interpolate_rsf — host serial reference vs parallel implementation
// ============================================================================
TEST_CASE("interpolate_rsf", "mo_photo") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("interpolate_rsf tests",
                                ekat::logger::LogLevel::debug, comm);

  constexpr int pver = mam4::nlev;
  constexpr Real Pa2mb = 1e-2;
  constexpr Real tol = PrecisionTolerance<Real>::tol;

  auto photo_table = build_test_photo_table();

  mam4::Atmosphere atm = mam4::init_atm_const_tv_lapse_rate(
      pver, 1000.0, 300.0, 0.01, 0.015, 7.5e-4);

  using View1D = mam4::mo_photo::View1D;
  using View2D = mam4::mo_photo::View2D;
  using View1DHost = Kokkos::View<Real *, Kokkos::HostSpace>;
  using View2DHost = Kokkos::View<Real **, Kokkos::HostSpace>;

  // Build alb_in, p_in, colo3_in inputs
  View1DHost alb_in_host("alb_in_h", pver);
  View1DHost p_in_host("p_in_h", pver);
  View1DHost colo3_in_host("colo3_in_h", pver);
  auto pressure_host = Kokkos::create_mirror_view(atm.pressure);
  Kokkos::deep_copy(pressure_host, atm.pressure);
  for (int k = 0; k < pver; ++k) {
    alb_in_host(k) = 0.15;
    p_in_host(k) = pressure_host(k) * Pa2mb;
    colo3_in_host(k) = 3e17;
  }
  View1D alb_in_d("alb_in", pver);
  View1D p_in_d("p_in", pver);
  View1D colo3_in_d("colo3_in", pver);
  Kokkos::deep_copy(alb_in_d, alb_in_host);
  Kokkos::deep_copy(p_in_d, p_in_host);
  Kokkos::deep_copy(colo3_in_d, colo3_in_host);

  // Mirror table arrays onto host for serial reference
  auto sza_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photo_table.sza);
  auto del_sza_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.del_sza);
  auto alb_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photo_table.alb);
  auto press_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.press);
  auto del_p_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.del_p);
  auto colo3_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.colo3);
  auto o3rat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.o3rat);
  auto del_alb_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.del_alb);
  auto del_o3rat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         photo_table.del_o3rat);
  auto etfphot_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.etfphot);
  auto rsf_tab_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.rsf_tab);

  // helper: find_index on host
  auto find_idx_h = [](const auto &arr, int len, Real val) {
    int idx = 0;
    for (int ii = 0; ii < len; ++ii) {
      if (arr(ii) > val) {
        idx = mam4::max(mam4::min(ii, len - 1) - 1, 0);
        break;
      }
    }
    return idx;
  };
  auto bound = [](Real lo, Real hi, Real v) {
    return v < lo ? lo : (v > hi ? hi : v);
  };

  // ===== SERIAL REFERENCE =====
  View2DHost rsf_ref("rsf_ref", test_nw, pver);

  // find zenith angle index and dels[0] — same for all levels
  int is_h = find_idx_h(sza_h, test_numsza, test_sza_in);
  Real dels0 = bound(0.0, 1.0, (test_sza_in - sza_h(is_h)) * del_sza_h(is_h));
  Real wrk0 = 1.0 - dels0;

  for (int kk = 0; kk < pver; ++kk) {
    int albind = find_idx_h(alb_h, test_numalb, alb_in_host(kk));

    int pind = 0;
    Real wght1 = 0;
    if (p_in_host(kk) > press_h(0)) {
      pind = 1;
      wght1 = 1.0;
    } else if (p_in_host(kk) <= press_h(test_nump - 1)) {
      pind = test_nump - 1;
      wght1 = 0.0;
    } else {
      int iz = 1;
      for (; iz < test_nump; ++iz)
        if (press_h(iz) < p_in_host(kk))
          break;
      iz = iz < test_nump - 1 ? iz : test_nump - 1;
      pind = iz > 1 ? iz : 1;
      wght1 =
          bound(0.0, 1.0, (p_in_host(kk) - press_h(pind)) * del_p_h(pind - 1));
    }

    Real v3ratu = colo3_in_host(kk) / colo3_h(pind - 1);
    int ratindu = find_idx_h(o3rat_h, test_numcolo3, v3ratu);

    Real v3ratl;
    int ratindl;
    if (colo3_h(pind) != 0.0) {
      v3ratl = colo3_in_host(kk) / colo3_h(pind);
      ratindl = find_idx_h(o3rat_h, test_numcolo3, v3ratl);
    } else {
      ratindl = ratindu;
      v3ratl = o3rat_h(ratindu);
    }

    Real dels2 =
        bound(0.0, 1.0, (alb_in_host(kk) - alb_h(albind)) * del_alb_h(albind));

    // calc_sum_wght for psum_l
    auto calc_psum = [&](int iz_c, int iv_c, Real v3rat) {
      Real dels1 = bound(0.0, 1.0, (v3rat - o3rat_h(iv_c)) * del_o3rat_h(iv_c));
      int is_c = is_h, isp1 = is_c + 1, ivp1 = iv_c + 1, ialp1 = albind + 1;
      Real wk = (1.0 - dels1) * (1.0 - dels2);
      Real w000 = wrk0 * wk, w100 = dels0 * wk;
      wk = (1.0 - dels1) * dels2;
      Real w001 = wrk0 * wk, w101 = dels0 * wk;
      wk = dels1 * (1.0 - dels2);
      Real w010 = wrk0 * wk, w110 = dels0 * wk;
      wk = dels1 * dels2;
      Real w011 = wrk0 * wk, w111 = dels0 * wk;
      Real psum[test_nw] = {};
      for (int wn = 0; wn < test_nw; ++wn)
        psum[wn] = w000 * rsf_tab_h(wn, iz_c, is_c, iv_c, albind) +
                   w001 * rsf_tab_h(wn, iz_c, is_c, iv_c, ialp1) +
                   w010 * rsf_tab_h(wn, iz_c, is_c, ivp1, albind) +
                   w011 * rsf_tab_h(wn, iz_c, is_c, ivp1, ialp1) +
                   w100 * rsf_tab_h(wn, iz_c, isp1, iv_c, albind) +
                   w101 * rsf_tab_h(wn, iz_c, isp1, iv_c, ialp1) +
                   w110 * rsf_tab_h(wn, iz_c, isp1, ivp1, albind) +
                   w111 * rsf_tab_h(wn, iz_c, isp1, ivp1, ialp1);
      // Copy into a fixed-size array for return by value
      struct PsumArr {
        Real v[test_nw];
      };
      PsumArr ret{};
      for (int wn = 0; wn < test_nw; ++wn)
        ret.v[wn] = psum[wn];
      return ret;
    };
    auto psum_l = calc_psum(pind, ratindl, v3ratl);
    auto psum_u = calc_psum(pind - 1, ratindu, v3ratu);

    for (int wn = 0; wn < test_nw; ++wn)
      rsf_ref(wn, kk) = (psum_l.v[wn] + wght1 * (psum_u.v[wn] - psum_l.v[wn])) *
                        etfphot_h(wn);
  }

  // ===== PARALLEL IMPLEMENTATION =====
  View2D rsf_par("rsf_par", test_nw, pver);
  {
    const auto sza_d = photo_table.sza;
    const auto del_sza_d = photo_table.del_sza;
    const auto alb_d = photo_table.alb;
    const auto press_d = photo_table.press;
    const auto del_p_d = photo_table.del_p;
    const auto colo3_d = photo_table.colo3;
    const auto o3rat_d = photo_table.o3rat;
    const auto del_alb_d = photo_table.del_alb;
    const auto del_o3rat_d = photo_table.del_o3rat;
    const auto etfphot_d = photo_table.etfphot;
    const auto rsf_tab_d = photo_table.rsf_tab;
    View2D psum_l_d("psum_l", pver, test_nw);
    View2D psum_u_d("psum_u", pver, test_nw);

    Kokkos::parallel_for(
        "interpolate_rsf_par", mam4::ThreadTeamPolicy(1, team_size),
        KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          mam4::mo_photo::interpolate_rsf(
              team, alb_in_d, test_sza_in, p_in_d, colo3_in_d, pver, sza_d,
              del_sza_d, alb_d, press_d, del_p_d, colo3_d, o3rat_d, del_alb_d,
              del_o3rat_d, etfphot_d, rsf_tab_d, test_nw, test_nump,
              test_numsza, test_numcolo3, test_numalb, rsf_par, psum_l_d,
              psum_u_d);
        });
    Kokkos::fence();
  }

  // ===== COMPARE =====
  auto rsf_par_h = Kokkos::create_mirror_view(rsf_par);
  Kokkos::deep_copy(rsf_par_h, rsf_par);
  for (int wn = 0; wn < test_nw; ++wn) {
    for (int k = 0; k < pver; ++k) {
      const Real ref = rsf_ref(wn, k);
      const Real par = rsf_par_h(wn, k);
      const Real aref = ref > 0 ? ref : -ref;
      const Real adiff = (par - ref) > 0 ? (par - ref) : -(par - ref);
      const Real diff = aref != 0.0 ? adiff / aref : adiff;
      if (diff > tol) {
        std::ostringstream ss;
        ss << "RSF mismatch wn=" << wn << " k=" << k << " parallel=" << par
           << " serial=" << ref << " diff=" << diff;
        logger.debug(ss.str());
      }
      REQUIRE(diff <= tol);
    }
  }
}
// ============================================================================
// Test: jlong — host serial reference vs parallel implementation
// ============================================================================
TEST_CASE("jlong", "mo_photo") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("jlong tests", ekat::logger::LogLevel::debug,
                                comm);

  constexpr int pver = mam4::nlev;
  constexpr Real Pa2mb = 1e-2;
  constexpr Real tol = PrecisionTolerance<Real>::tol;

  auto photo_table = build_test_photo_table();

  mam4::Atmosphere atm = mam4::init_atm_const_tv_lapse_rate(
      pver, 1000.0, 300.0, 0.01, 0.015, 7.5e-4);

  using View1D = mam4::mo_photo::View1D;
  using View2D = mam4::mo_photo::View2D;
  using View3D = mam4::mo_photo::View3D;
  using View1DHost = Kokkos::View<Real *, Kokkos::HostSpace>;
  using View2DHost = Kokkos::View<Real **, Kokkos::HostSpace>;

  View1DHost alb_in_host("alb_in_h", pver);
  View1DHost p_in_host("p_in_h", pver);
  View1DHost colo3_in_host("colo3_in_h", pver);
  auto pressure_host = Kokkos::create_mirror_view(atm.pressure);
  Kokkos::deep_copy(pressure_host, atm.pressure);
  auto temper_host = Kokkos::create_mirror_view(atm.temperature);
  Kokkos::deep_copy(temper_host, atm.temperature);
  for (int k = 0; k < pver; ++k) {
    alb_in_host(k) = 0.15;
    p_in_host(k) = pressure_host(k) * Pa2mb;
    colo3_in_host(k) = 3e17;
  }
  View1D alb_in_d("alb_in", pver);
  View1D p_in_d("p_in", pver);
  View1D colo3_in_d("colo3_in", pver);
  Kokkos::deep_copy(alb_in_d, alb_in_host);
  Kokkos::deep_copy(p_in_d, p_in_host);
  Kokkos::deep_copy(colo3_in_d, colo3_in_host);

  // Mirror table onto host
  auto sza_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photo_table.sza);
  auto del_sza_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.del_sza);
  auto alb_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photo_table.alb);
  auto press_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.press);
  auto del_p_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.del_p);
  auto colo3_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.colo3);
  auto o3rat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     photo_table.o3rat);
  auto del_alb_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.del_alb);
  auto del_o3rat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         photo_table.del_o3rat);
  auto etfphot_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.etfphot);
  auto rsf_tab_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       photo_table.rsf_tab);
  auto xsqy_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                    photo_table.xsqy);
  auto prs_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, photo_table.prs);
  auto dprs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                    photo_table.dprs);

  auto find_idx_h = [](const auto &arr, int len, Real val) {
    int idx = 0;
    for (int ii = 0; ii < len; ++ii) {
      if (arr(ii) > val) {
        idx = mam4::max(mam4::min(ii, len - 1) - 1, 0);
        break;
      }
    }
    return idx;
  };
  auto bound = [](Real lo, Real hi, Real v) {
    return v < lo ? lo : (v > hi ? hi : v);
  };

  // ===== SERIAL REFERENCE =====
  // Step 1: compute rsf_ref using same logic as interpolate_rsf
  View2DHost rsf_ref("rsf_ref", test_nw, pver);
  int is_h = find_idx_h(sza_h, test_numsza, test_sza_in);
  Real dels0 = bound(0.0, 1.0, (test_sza_in - sza_h(is_h)) * del_sza_h(is_h));
  Real wrk0 = 1.0 - dels0;

  for (int kk = 0; kk < pver; ++kk) {
    int albind = find_idx_h(alb_h, test_numalb, alb_in_host(kk));
    int pind = 0;
    Real wght1 = 0;
    if (p_in_host(kk) > press_h(0)) {
      pind = 1;
      wght1 = 1.0;
    } else if (p_in_host(kk) <= press_h(test_nump - 1)) {
      pind = test_nump - 1;
      wght1 = 0.0;
    } else {
      int iz = 1;
      for (; iz < test_nump; ++iz)
        if (press_h(iz) < p_in_host(kk))
          break;
      iz = iz < test_nump - 1 ? iz : test_nump - 1;
      pind = iz > 1 ? iz : 1;
      wght1 =
          bound(0.0, 1.0, (p_in_host(kk) - press_h(pind)) * del_p_h(pind - 1));
    }
    Real v3ratu = colo3_in_host(kk) / colo3_h(pind - 1);
    int ratindu = find_idx_h(o3rat_h, test_numcolo3, v3ratu);
    Real v3ratl;
    int ratindl;
    if (colo3_h(pind) != 0.0) {
      v3ratl = colo3_in_host(kk) / colo3_h(pind);
      ratindl = find_idx_h(o3rat_h, test_numcolo3, v3ratl);
    } else {
      ratindl = ratindu;
      v3ratl = o3rat_h(ratindu);
    }
    Real dels2 =
        bound(0.0, 1.0, (alb_in_host(kk) - alb_h(albind)) * del_alb_h(albind));
    int ialp1 = albind + 1;

    auto calc_psum = [&](int iz_c, int iv_c, Real v3rat) {
      Real dels1 = bound(0.0, 1.0, (v3rat - o3rat_h(iv_c)) * del_o3rat_h(iv_c));
      int isp1 = is_h + 1, ivp1 = iv_c + 1;
      Real wk = (1.0 - dels1) * (1.0 - dels2);
      Real w000 = wrk0 * wk, w100 = dels0 * wk;
      wk = (1.0 - dels1) * dels2;
      Real w001 = wrk0 * wk, w101 = dels0 * wk;
      wk = dels1 * (1.0 - dels2);
      Real w010 = wrk0 * wk, w110 = dels0 * wk;
      wk = dels1 * dels2;
      Real w011 = wrk0 * wk, w111 = dels0 * wk;
      struct PsArr {
        Real v[test_nw];
      };
      PsArr ps{};
      for (int wn = 0; wn < test_nw; ++wn)
        ps.v[wn] = w000 * rsf_tab_h(wn, iz_c, is_h, iv_c, albind) +
                   w001 * rsf_tab_h(wn, iz_c, is_h, iv_c, ialp1) +
                   w010 * rsf_tab_h(wn, iz_c, is_h, ivp1, albind) +
                   w011 * rsf_tab_h(wn, iz_c, is_h, ivp1, ialp1) +
                   w100 * rsf_tab_h(wn, iz_c, isp1, iv_c, albind) +
                   w101 * rsf_tab_h(wn, iz_c, isp1, iv_c, ialp1) +
                   w110 * rsf_tab_h(wn, iz_c, isp1, ivp1, albind) +
                   w111 * rsf_tab_h(wn, iz_c, isp1, ivp1, ialp1);
      return ps;
    };
    auto psum_l = calc_psum(pind, ratindl, v3ratl);
    auto psum_u = calc_psum(pind - 1, ratindu, v3ratu);
    for (int wn = 0; wn < test_nw; ++wn)
      rsf_ref(wn, kk) = (psum_l.v[wn] + wght1 * (psum_u.v[wn] - psum_l.v[wn])) *
                        etfphot_h(wn);
  }

  // Step 2: compute j_long_ref on host
  View2DHost j_long_ref("j_long_ref", test_numj, pver);
  for (int kk = 0; kk < pver; ++kk) {
    const int ti0 = (int)(temper_host(kk) - 148.5);
    const int t_index = (ti0 < 1 ? 1 : (ti0 > 201 ? 201 : ti0)) - 1;

    Real ptarget = p_in_host(kk);
    Real xswk_row[test_nw] = {};
    if (ptarget >= prs_h(0)) {
      for (int wn = 0; wn < test_nw; ++wn)
        for (int i = 0; i < test_numj; ++i)
          xswk_row[wn] = xsqy_h(i, wn, t_index, 0);
    } else if (ptarget <= prs_h(test_np_xs - 1)) {
      for (int wn = 0; wn < test_nw; ++wn)
        for (int i = 0; i < test_numj; ++i)
          xswk_row[wn] = xsqy_h(i, wn, t_index, test_np_xs - 1);
    } else {
      Real delp = 0;
      int pndx = 0;
      for (int km = 1; km < test_np_xs; ++km) {
        if (ptarget >= prs_h(km)) {
          pndx = km - 1;
          delp = (prs_h(pndx) - ptarget) * dprs_h(pndx);
          break;
        }
      }
      for (int wn = 0; wn < test_nw; ++wn)
        for (int i = 0; i < test_numj; ++i)
          xswk_row[wn] = xsqy_h(i, wn, t_index, pndx) +
                         delp * (xsqy_h(i, wn, t_index, pndx + 1) -
                                 xsqy_h(i, wn, t_index, pndx));
    }
    for (int i = 0; i < test_numj; ++i) {
      Real suma = 0;
      for (int wn = 0; wn < test_nw; ++wn)
        suma += xswk_row[wn] * rsf_ref(wn, kk);
      j_long_ref(i, kk) = suma;
    }
  }

  // ===== PARALLEL IMPLEMENTATION =====
  View2D j_long_par("j_long_par", test_numj, pver);
  {
    View2D rsf_work("rsf_work", test_nw, pver);
    View2D psum_l_d("psum_l", pver, test_nw);
    View2D psum_u_d("psum_u", pver, test_nw);

    const auto sza_d = photo_table.sza;
    const auto del_sza_d = photo_table.del_sza;
    const auto alb_d = photo_table.alb;
    const auto press_d = photo_table.press;
    const auto del_p_d = photo_table.del_p;
    const auto colo3_d = photo_table.colo3;
    const auto o3rat_d = photo_table.o3rat;
    const auto del_alb_d = photo_table.del_alb;
    const auto del_o3rat_d = photo_table.del_o3rat;
    const auto etfphot_d = photo_table.etfphot;
    const auto rsf_tab_d = photo_table.rsf_tab;
    const auto xsqy_d = photo_table.xsqy;
    const auto prs_d = photo_table.prs;
    const auto dprs_d = photo_table.dprs;
    const auto temper = atm.temperature;

    Kokkos::parallel_for(
        "jlong_par", mam4::ThreadTeamPolicy(1, team_size),
        KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          mam4::mo_photo::jlong(
              team, test_sza_in, alb_in_d, p_in_d, temper, colo3_in_d, xsqy_d,
              sza_d, del_sza_d, alb_d, press_d, del_p_d, colo3_d, o3rat_d,
              del_alb_d, del_o3rat_d, etfphot_d, rsf_tab_d, prs_d, dprs_d,
              test_nw, test_nump, test_numsza, test_numcolo3, test_numalb,
              test_np_xs, test_numj, j_long_par, rsf_work, psum_l_d,
              psum_u_d);
        });
    Kokkos::fence();
  }

  // ===== COMPARE =====
  auto j_par_h = Kokkos::create_mirror_view(j_long_par);
  Kokkos::deep_copy(j_par_h, j_long_par);
  for (int i = 0; i < test_numj; ++i) {
    for (int k = 0; k < pver; ++k) {
      const Real ref = j_long_ref(i, k);
      const Real par = j_par_h(i, k);
      const Real aref2 = ref > 0 ? ref : -ref;
      const Real adiff2 = (par - ref) > 0 ? (par - ref) : -(par - ref);
      const Real diff = aref2 != 0.0 ? adiff2 / aref2 : adiff2;
      if (diff > tol) {
        std::ostringstream ss;
        ss << "j_long mismatch reaction=" << i << " k=" << k
           << " parallel=" << par << " serial=" << ref << " diff=" << diff;
        logger.debug(ss.str());
      }
      REQUIRE(diff <= tol);
    }
  }
}
