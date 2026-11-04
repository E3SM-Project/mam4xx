// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

// Unit tests for the optional implicit vertical-mixing integrators in
// ndrop::update_from_explmix (E3SM docs/design/eamxx_mam4xx_ndrop.md,
// "Production Implicit Vertical-Mixing Option", verification plan items 1-5).
//
// Host-side oracles are assembled directly from the physical equations (not
// from the production block bands) and solved with dense partial-pivoting
// Gaussian elimination. The temporal-accuracy reference is a converged RK4
// integration of the exact frozen linear operator, verified by step halving.

#include <mam4xx/mam4.hpp>

#include "testing.hpp"

#include <catch2/catch.hpp>
#include <ekat_comm.hpp>
#include <ekat_logger.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using mam4::Real;
using mam4::ndrop::ImplicitSubstepPolicy;
using mam4::ndrop::ImplicitWorkspace;
using mam4::ndrop::MixingIntegrator;
using mam4::ndrop::MixingOptions;
using mam4::ndrop::MixingStatus;

#ifdef MAM4XX_ENABLE_GPU
constexpr int team_size = mam4::nlev;
#else
constexpr int team_size = 1;
#endif

namespace {

constexpr int pver = mam4::ndrop::pver;
constexpr int nmodes = mam4::AeroConfig::num_modes();
constexpr int ncnst = mam4::ndrop::ncnst_tot;
constexpr double gravity = 9.80616;

// ---------------------------------------------------------------------------
// Host description of one column case
// ---------------------------------------------------------------------------

struct HostCase {
  int top_lev = 0;
  double dtmicro = 1800.0;
  std::vector<double> zn, zs, csbot, eddy_diff, cldn, qcld;
  std::vector<std::vector<double>> nact, mact; // (pver, nmodes), uncapped
  std::vector<std::vector<double>> raercol, raercol_cw; // (pver, ncnst)
  HostCase()
      : zn(pver), zs(pver), csbot(pver), eddy_diff(pver), cldn(pver),
        qcld(pver), nact(pver, std::vector<double>(nmodes, 0.0)),
        mact(pver, std::vector<double>(nmodes, 0.0)),
        raercol(pver, std::vector<double>(ncnst, 0.0)),
        raercol_cw(pver, std::vector<double>(ncnst, 0.0)) {}
};

// A boundary-layer-like column: uniform 80 m layers, air density 1 kg/m3,
// a cloud slab, mixing K [m2/s] in the active range with a cloud-base
// override, activation at the cloud-base layers, and smooth positive aerosol
// profiles. kind: "stiff" (strong mixing), "limiter" (activation rates far
// above the cap so the cap binds), "two_level" (top_lev = pver-2, used to
// force an IMEX material negative).
HostCase make_case(const std::string &kind, const int top_lev_in) {
  HostCase c;
  c.top_lev = top_lev_in;
  const double dz = 80.0;
  const double rho = 1.0;
  const double pdel = rho * gravity * dz; // hydrostatic layer thickness [Pa]
  // kind: stiff/limiter -> strong mixing (dtmicro/dtmin >= 50); mild -> weak
  // mixing and a short interval so a single IMEX step is accepted
  const double K = (kind == "two_level") ? 10.0 : (kind == "mild" ? 3.0 : 30.0);
  if (kind == "mild")
    c.dtmicro = 300.0;
  const int cloud_top = pver - 24, cloud_base = pver - 14; // slab indices
  for (int k = 0; k < pver; ++k) {
    c.zn[k] = gravity / pdel;
    c.zs[k] = 1.0 / dz;
    c.csbot[k] = rho;
    c.eddy_diff[k] = K;
    const bool cloudy = (k >= cloud_top && k <= cloud_base);
    c.cldn[k] = cloudy ? 0.9 : 0.2; // strictly positive: no zero-cloud reset
    c.qcld[k] = cloudy ? 5.0e7 : 1.0e6;
    for (int mm = 0; mm < ncnst; ++mm) {
      const double prof = 1.0 + 0.5 * (double(k) / double(pver - 1));
      const bool is_number = (mm == 0 || mm == 8 || mm == 13 || mm == 21);
      const double base = is_number ? 1.0e9 : 1.0e-9;
      c.raercol[k][mm] = base * prof * (1.0 + 0.01 * mm);
      c.raercol_cw[k][mm] =
          cloudy ? 0.3 * base * prof * (1.0 + 0.01 * mm) : 0.0;
    }
  }
  // surface cell as dropmixnuc sets it: no exchange below the bottom level
  c.eddy_diff[pver - 1] = 0.0;
  // cloud-base override (eddy_diff = wtke/zs) and activation at the layers
  // just above cloud base; activation is zero at the surface cell
  const double wtke = (kind == "mild") ? 0.3 : 1.2;
  for (int k = cloud_base - 2; k <= cloud_base; ++k) {
    if (k < 0 || k >= pver - 1)
      continue;
    c.eddy_diff[k] = wtke / c.zs[k];
    for (int m = 0; m < nmodes; ++m) {
      const double rate = c.zn[k] * c.eddy_diff[k] * c.csbot[k] * c.zs[k];
      const double frac = (kind == "limiter") ? 50.0 : 0.4;
      c.nact[k][m] = frac * rate;
      c.mact[k][m] = frac * 0.8 * rate;
    }
  }
  if (kind == "two_level") {
    // two active levels: top = pver-2 with activation at the cap, bottom
    // surface cell with none; interstitial zero at the top, large below
    c.top_lev = pver - 2;
    const int k = pver - 2;
    c.eddy_diff[k] = K;
    c.cldn[k] = 0.9;
    c.cldn[pver - 1] = 0.9;
    for (int m = 0; m < nmodes; ++m) {
      c.nact[k][m] = 1.0e3; // far above the cap: capped to eddy_diff_kp
      c.mact[k][m] = 1.0e3;
      c.nact[pver - 1][m] = 0.0;
      c.mact[pver - 1][m] = 0.0;
    }
    for (int mm = 0; mm < ncnst; ++mm) {
      c.raercol[k][mm] = 0.0;
      c.raercol_cw[k][mm] = 0.0;
      c.raercol_cw[pver - 1][mm] = 0.0;
    }
    c.qcld[k] = 0.0;
    c.qcld[pver - 1] = 0.0;
  }
  return c;
}

// ---------------------------------------------------------------------------
// Host replica of the production operator setup (rates, overlaps, cap,
// controller), used by every oracle
// ---------------------------------------------------------------------------

struct HostOperator {
  int top_lev = 0;
  std::vector<double> a, b, oA, oB, w; // a: rate toward k-1 (excluded at top)
  std::vector<std::vector<double>> act_n, act_m; // capped
  double dtmin = 0.0;
  int nsubmix(const double dtmicro) const {
    const double dtmix = 0.9 * dtmin;
    return static_cast<int>(dtmicro / dtmix) + 1;
  }
};

HostOperator make_operator(const HostCase &c) {
  HostOperator op;
  op.top_lev = c.top_lev;
  op.a.assign(pver, 0.0);
  op.b.assign(pver, 0.0);
  op.oA.assign(pver, 1.0);
  op.oB.assign(pver, 1.0);
  op.w.assign(pver, 0.0);
  op.act_n.assign(pver, std::vector<double>(nmodes, 0.0));
  op.act_m.assign(pver, std::vector<double>(nmodes, 0.0));
  op.dtmin = c.dtmicro;
  for (int k = c.top_lev; k < pver; ++k) {
    const int kp1 = std::min(k + 1, pver - 1);
    const int km1 = std::max(k - 1, c.top_lev);
    op.oB[k] =
        (c.cldn[kp1] > 1e-10) ? std::min(c.cldn[k] / c.cldn[kp1], 1.0) : 1.0;
    op.oA[k] =
        (c.cldn[km1] > 1e-10) ? std::min(c.cldn[k] / c.cldn[km1], 1.0) : 1.0;
    const double kp = c.zn[k] * c.eddy_diff[k] * c.csbot[k] * c.zs[k];
    const double km = c.zn[k] * c.eddy_diff[km1] * c.csbot[km1] * c.zs[km1];
    for (int m = 0; m < nmodes; ++m) {
      op.act_n[k][m] = std::min(c.nact[k][m], kp);
      op.act_m[k][m] = std::min(c.mact[k][m], kp);
    }
    const double tinv = kp + km;
    if (tinv > 1e-6)
      op.dtmin = std::min(op.dtmin, 1.0 / tinv);
    op.a[k] = (k == c.top_lev) ? 0.0 : km;
    op.b[k] = (k == pver - 1) ? 0.0 : kp;
    op.w[k] = 1.0 / c.zn[k];
  }
  return op;
}

// constituent -> (mode, is_mass)
void constituent_mode(const int mm, int &imode, bool &is_mass) {
  for (int m = 0; m < nmodes; ++m) {
    for (int l = 0; l < mam4::AeroConfig::nspec_amode(m) + 1; ++l) {
      if (mam4::AeroConfig::mam_idx(m, l) == mm) {
        imode = m;
        is_mass = (l > 0);
        return;
      }
    }
  }
  REQUIRE(false);
}

double act_of(const HostOperator &op, const int k, const int mm) {
  int m;
  bool is_mass;
  constituent_mode(mm, m, is_mass);
  return is_mass ? op.act_m[k][m] : op.act_n[k][m];
}

// ---------------------------------------------------------------------------
// Dense linear algebra
// ---------------------------------------------------------------------------

using Mat = std::vector<std::vector<double>>;

std::vector<double> solve_dense(Mat A, std::vector<double> rhs) {
  const int n = static_cast<int>(rhs.size());
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r)
      if (std::fabs(A[r][col]) > std::fabs(A[piv][col]))
        piv = r;
    REQUIRE(std::fabs(A[piv][col]) > 0.0);
    std::swap(A[piv], A[col]);
    std::swap(rhs[piv], rhs[col]);
    for (int r = col + 1; r < n; ++r) {
      const double f = A[r][col] / A[col][col];
      if (f == 0.0)
        continue;
      for (int cc = col; cc < n; ++cc)
        A[r][cc] -= f * A[col][cc];
      rhs[r] -= f * rhs[col];
    }
  }
  std::vector<double> x(n, 0.0);
  for (int r = n - 1; r >= 0; --r) {
    double s = rhs[r];
    for (int cc = r + 1; cc < n; ++cc)
      s -= A[r][cc] * x[cc];
    x[r] = s / A[r][r];
  }
  return x;
}

// ---------------------------------------------------------------------------
// Host oracles over the active range [top_lev, pver)
// ---------------------------------------------------------------------------

struct HostState {
  std::vector<double> N;                 // droplets (pver)
  std::vector<std::vector<double>> C, I; // (pver, ncnst)
};

HostState host_state(const HostCase &c) {
  HostState s;
  s.N = c.qcld;
  s.C = c.raercol_cw;
  s.I = c.raercol;
  return s;
}

// scalar family matrices (dense) over active levels
Mat scalar_matrix(const HostOperator &op, const double h, const bool cloud) {
  const int t = op.top_lev, n = pver - t;
  Mat A(n, std::vector<double>(n, 0.0));
  for (int k = t; k < pver; ++k) {
    const int r = k - t;
    A[r][r] = 1.0 + h * (op.a[k] + op.b[k]);
    if (k > t)
      A[r][r - 1] = -h * op.a[k] * (cloud ? op.oA[k] : 1.0);
    if (k < pver - 1)
      A[r][r + 1] = -h * op.b[k] * (cloud ? op.oB[k] : 1.0);
  }
  return A;
}

// IMEX-Euler step of all 51 fields
HostState imex_oracle_step(const HostOperator &op, const HostState &s,
                           const double h) {
  const int t = op.top_lev, n = pver - t;
  HostState out = s;
  const Mat AC = scalar_matrix(op, h, true);
  const Mat AI = scalar_matrix(op, h, false);
  // droplets: explicit source from old interstitial numbers
  {
    std::vector<double> rhs(n);
    for (int k = t; k < pver; ++k) {
      const int kp1 = std::min(k + 1, pver - 1);
      double srcn = 0.0;
      for (int m = 0; m < nmodes; ++m)
        srcn += op.act_n[k][m] * s.I[kp1][mam4::AeroConfig::mam_idx(m, 0)];
      rhs[k - t] = s.N[k] + h * srcn;
    }
    const auto x = solve_dense(AC, rhs);
    for (int k = t; k < pver; ++k)
      out.N[k] = x[k - t];
  }
  for (int mm = 0; mm < ncnst; ++mm) {
    std::vector<double> rhs(n);
    for (int k = t; k < pver; ++k) {
      const int kp1 = std::min(k + 1, pver - 1);
      rhs[k - t] = s.C[k][mm] + h * act_of(op, k, mm) * s.I[kp1][mm];
    }
    const auto Cn = solve_dense(AC, rhs);
    for (int k = t; k < pver; ++k) {
      const int kp1 = std::min(k + 1, pver - 1);
      const int km1 = std::max(k - 1, t);
      const double S = act_of(op, k, mm) * s.I[kp1][mm];
      rhs[k - t] = s.I[k][mm] - h * S +
                   h * op.a[k] * (1.0 - op.oA[k]) * Cn[km1 - t] +
                   h * op.b[k] * (1.0 - op.oB[k]) * Cn[kp1 - t];
    }
    const auto In = solve_dense(AI, rhs);
    for (int k = t; k < pver; ++k) {
      out.C[k][mm] = Cn[k - t];
      out.I[k][mm] = In[k - t];
    }
  }
  return out;
}

// Backward Euler step: 2n-by-2n per constituent from the physical equations,
// then droplets with the solved interstitial numbers
HostState be_oracle_step(const HostOperator &op, const HostState &s,
                         const double h) {
  const int t = op.top_lev, n = pver - t;
  HostState out = s;
  for (int mm = 0; mm < ncnst; ++mm) {
    Mat A(2 * n, std::vector<double>(2 * n, 0.0));
    std::vector<double> rhs(2 * n);
    for (int k = t; k < pver; ++k) {
      const int r = k - t;
      const int kp1 = std::min(k + 1, pver - 1);
      const int km1 = std::max(k - 1, t);
      const double act = act_of(op, k, mm);
      // unknown ordering: C_r at 2r, I_r at 2r+1
      const int iC = 2 * r, iI = 2 * r + 1;
      // C_k - h[ a oA C_{k-1} + b oB C_{k+1} - (a+b) C_k + act I_{k+1} ] = C^n
      A[iC][iC] += 1.0 + h * (op.a[k] + op.b[k]);
      if (k > t)
        A[iC][2 * (km1 - t)] += -h * op.a[k] * op.oA[k];
      if (k < pver - 1) {
        A[iC][2 * (kp1 - t)] += -h * op.b[k] * op.oB[k];
        A[iC][2 * (kp1 - t) + 1] += -h * act;
      }
      rhs[iC] = s.C[k][mm];
      // I_k - h[ a I_{k-1} + b I_{k+1} - (a+b) I_k + a(1-oA) C_{k-1}
      //         + b(1-oB) C_{k+1} - act I_{k+1} ] = I^n
      A[iI][iI] += 1.0 + h * (op.a[k] + op.b[k]);
      if (k > t) {
        A[iI][2 * (km1 - t) + 1] += -h * op.a[k];
        A[iI][2 * (km1 - t)] += -h * op.a[k] * (1.0 - op.oA[k]);
      }
      if (k < pver - 1) {
        A[iI][2 * (kp1 - t) + 1] += -h * op.b[k] + h * act;
        A[iI][2 * (kp1 - t)] += -h * op.b[k] * (1.0 - op.oB[k]);
      }
      rhs[iI] = s.I[k][mm];
    }
    const auto x = solve_dense(A, rhs);
    for (int k = t; k < pver; ++k) {
      out.C[k][mm] = x[2 * (k - t)];
      out.I[k][mm] = x[2 * (k - t) + 1];
    }
  }
  {
    const Mat AC = scalar_matrix(op, h, true);
    std::vector<double> rhs(n);
    for (int k = t; k < pver; ++k) {
      const int kp1 = std::min(k + 1, pver - 1);
      double srcn = 0.0;
      for (int m = 0; m < nmodes; ++m)
        srcn += op.act_n[k][m] * out.I[kp1][mam4::AeroConfig::mam_idx(m, 0)];
      rhs[k - t] = s.N[k] + h * srcn;
    }
    const auto x = solve_dense(AC, rhs);
    for (int k = t; k < pver; ++k)
      out.N[k] = x[k - t];
  }
  return out;
}

// Exact frozen linear operator (right-hand side of dz/dt = M z) on all fields
HostState exact_rhs(const HostOperator &op, const HostState &s) {
  const int t = op.top_lev;
  HostState d = s;
  for (int k = 0; k < t; ++k) {
    d.N[k] = 0.0;
    for (int mm = 0; mm < ncnst; ++mm)
      d.C[k][mm] = d.I[k][mm] = 0.0;
  }
  for (int k = t; k < pver; ++k) {
    const int kp1 = std::min(k + 1, pver - 1);
    const int km1 = std::max(k - 1, t);
    const double a = op.a[k], b = op.b[k], oA = op.oA[k], oB = op.oB[k];
    double srcn = 0.0;
    for (int m = 0; m < nmodes; ++m)
      srcn += op.act_n[k][m] * s.I[kp1][mam4::AeroConfig::mam_idx(m, 0)];
    d.N[k] = srcn + b * (oB * s.N[kp1] - s.N[k]) + a * (oA * s.N[km1] - s.N[k]);
    for (int mm = 0; mm < ncnst; ++mm) {
      const double S = act_of(op, k, mm) * s.I[kp1][mm];
      d.C[k][mm] = S + b * (oB * s.C[kp1][mm] - s.C[k][mm]) +
                   a * (oA * s.C[km1][mm] - s.C[k][mm]);
      d.I[k][mm] = -S +
                   b * (s.I[kp1][mm] - s.I[k][mm] + (1.0 - oB) * s.C[kp1][mm]) +
                   a * (s.I[km1][mm] - s.I[k][mm] + (1.0 - oA) * s.C[km1][mm]);
    }
  }
  return d;
}

HostState axpy(const HostState &x, const double alpha, const HostState &y) {
  HostState r = x;
  for (int k = 0; k < pver; ++k) {
    r.N[k] += alpha * y.N[k];
    for (int mm = 0; mm < ncnst; ++mm) {
      r.C[k][mm] += alpha * y.C[k][mm];
      r.I[k][mm] += alpha * y.I[k][mm];
    }
  }
  return r;
}

// RK4 on the exact linear operator with nsteps steps over T
HostState rk4_reference(const HostOperator &op, const HostState &s0,
                        const double T, const int nsteps) {
  HostState s = s0;
  const double h = T / nsteps;
  for (int i = 0; i < nsteps; ++i) {
    const HostState k1 = exact_rhs(op, s);
    const HostState k2 = exact_rhs(op, axpy(s, 0.5 * h, k1));
    const HostState k3 = exact_rhs(op, axpy(s, 0.5 * h, k2));
    const HostState k4 = exact_rhs(op, axpy(s, h, k3));
    HostState incr = axpy(k1, 2.0, k2);
    incr = axpy(incr, 2.0, k3);
    incr = axpy(incr, 1.0, k4);
    s = axpy(s, h / 6.0, incr);
  }
  return s;
}

// maximum over the 51 fields of the normalized L-infinity difference
double max_normalized_diff(const HostOperator &op, const HostState &x,
                           const HostState &ref) {
  const int t = op.top_lev;
  double worst = 0.0;
  auto field = [&](auto getx, auto getref) {
    double scale = 0.0, diff = 0.0;
    for (int k = t; k < pver; ++k) {
      scale = std::max(scale, std::fabs(getref(k)));
      diff = std::max(diff, std::fabs(getx(k) - getref(k)));
    }
    if (scale > 0.0)
      worst = std::max(worst, diff / scale);
    else
      worst = std::max(worst, diff);
  };
  field([&](int k) { return x.N[k]; }, [&](int k) { return ref.N[k]; });
  for (int mm = 0; mm < ncnst; ++mm) {
    field([&](int k) { return x.C[k][mm]; },
          [&](int k) { return ref.C[k][mm]; });
    field([&](int k) { return x.I[k][mm]; },
          [&](int k) { return ref.I[k][mm]; });
  }
  return worst;
}

// relative ledger change per aerosol constituent (both phases). Droplet
// number is not conserved: activation is a source for it by design.
double max_relative_ledger_change(const HostOperator &op, const HostState &x,
                                  const HostState &x0) {
  const int t = op.top_lev;
  double worst = 0.0;
  auto ledger = [&](const HostState &s, int mm) {
    double L = 0.0;
    for (int k = t; k < pver; ++k)
      L += op.w[k] * (s.C[k][mm] + s.I[k][mm]);
    return L;
  };
  for (int mm = 0; mm < ncnst; ++mm) {
    const double L0 = ledger(x0, mm), L1 = ledger(x, mm);
    if (L0 > 0.0)
      worst = std::max(worst, std::fabs(L1 - L0) / L0);
    else
      worst = std::max(worst, std::fabs(L1 - L0));
  }
  return worst;
}

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------

struct DeviceCase {
  mam4::ColumnView csbot, cldn, zn, zs, eddy_diff, qcld, overlapp, overlapm,
      eddy_diff_kp, eddy_diff_km, qncld;
  mam4::ndrop::View2D nact, mact;
  mam4::ndrop::View3D raercol, raercol_cw;
};

DeviceCase to_device(const HostCase &c) {
  DeviceCase d;
  auto col = [](const std::vector<double> &v, mam4::ColumnView &out) {
    out = mam4::testing::create_column_view(pver);
    auto h = Kokkos::create_mirror_view(out);
    for (int k = 0; k < pver; ++k)
      h(k) = static_cast<Real>(v[k]);
    Kokkos::deep_copy(out, h);
  };
  col(c.csbot, d.csbot);
  col(c.cldn, d.cldn);
  col(c.zn, d.zn);
  col(c.zs, d.zs);
  col(c.eddy_diff, d.eddy_diff);
  col(c.qcld, d.qcld);
  d.overlapp = mam4::testing::create_column_view(pver);
  d.overlapm = mam4::testing::create_column_view(pver);
  d.eddy_diff_kp = mam4::testing::create_column_view(pver);
  d.eddy_diff_km = mam4::testing::create_column_view(pver);
  d.qncld = mam4::testing::create_column_view(pver);
  d.nact = mam4::ndrop::View2D("nact", pver, nmodes);
  d.mact = mam4::ndrop::View2D("mact", pver, nmodes);
  d.raercol = mam4::ndrop::View3D("raercol", pver, 2, ncnst);
  d.raercol_cw = mam4::ndrop::View3D("raercol_cw", pver, 2, ncnst);
  auto hn = Kokkos::create_mirror_view(d.nact);
  auto hm = Kokkos::create_mirror_view(d.mact);
  auto hr = Kokkos::create_mirror_view(d.raercol);
  auto hc = Kokkos::create_mirror_view(d.raercol_cw);
  for (int k = 0; k < pver; ++k) {
    for (int m = 0; m < nmodes; ++m) {
      hn(k, m) = static_cast<Real>(c.nact[k][m]);
      hm(k, m) = static_cast<Real>(c.mact[k][m]);
    }
    for (int mm = 0; mm < ncnst; ++mm) {
      hr(k, 0, mm) = static_cast<Real>(c.raercol[k][mm]);
      hc(k, 0, mm) = static_cast<Real>(c.raercol_cw[k][mm]);
      hr(k, 1, mm) = Real(-7); // sentinel: never read before written
      hc(k, 1, mm) = Real(-7);
    }
  }
  Kokkos::deep_copy(d.nact, hn);
  Kokkos::deep_copy(d.mact, hm);
  Kokkos::deep_copy(d.raercol, hr);
  Kokkos::deep_copy(d.raercol_cw, hc);
  return d;
}

struct RunResult {
  HostState state;
  MixingStatus status;
};

// Run update_from_explmix nsplit times over dtmicro/nsplit each, with the
// result always left in buffer slot 0. Status counters are accumulated.
RunResult run_device(const DeviceCase &d, const int top_lev,
                     const double dtmicro, const int nsplit,
                     const MixingOptions opts, const ImplicitWorkspace ws) {
  mam4::DeviceType::view_1d<int> status_dev("status", 4);
  Kokkos::deep_copy(status_dev, 0);
  const Real dt = static_cast<Real>(dtmicro / nsplit);
  auto csbot = d.csbot, cldn = d.cldn, zn = d.zn, zs = d.zs,
       eddy_diff = d.eddy_diff, qcld = d.qcld, overlapp = d.overlapp,
       overlapm = d.overlapm, eddy_diff_kp = d.eddy_diff_kp,
       eddy_diff_km = d.eddy_diff_km, qncld = d.qncld;
  auto nact = d.nact, mact = d.mact;
  auto raercol = d.raercol, raercol_cw = d.raercol_cw;
  auto policy = mam4::ThreadTeamPolicy(1u, team_size);
  Kokkos::parallel_for(
      "implicit_mixing_run", policy,
      KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
        for (int split = 0; split < nsplit; ++split) {
          int nsav = 0;
          int nnew = 1;
#ifdef MAM4XX_EVAL_DIAGNOSTICS
          int nsubmix_out = 0;
#endif
          MixingStatus st;
          mam4::ndrop::update_from_explmix(
              team, dt, csbot, cldn, zn, zs, eddy_diff, nact, mact, qcld,
              raercol, raercol_cw, nsav, nnew,
#ifdef MAM4XX_EVAL_DIAGNOSTICS
              nsubmix_out,
#endif
              top_lev, overlapp, overlapm, eddy_diff_kp, eddy_diff_km, qncld,
              opts, ws, &st);
          team.team_barrier();
          if (nnew != 0) {
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, pver), [&](int k) {
                  for (int mm = 0; mm < ncnst; ++mm) {
                    raercol(k, 0, mm) = raercol(k, nnew, mm);
                    raercol_cw(k, 0, mm) = raercol_cw(k, nnew, mm);
                  }
                });
            team.team_barrier();
          }
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            status_dev(0) += st.nsubmix_executed;
            status_dev(1) += st.fallback_count;
            status_dev(2) += st.material_negative_count;
            status_dev(3) = st.input_negative_count;
          });
          team.team_barrier();
        }
      });
  Kokkos::fence();
  RunResult r;
  auto hs = Kokkos::create_mirror_view(status_dev);
  Kokkos::deep_copy(hs, status_dev);
  r.status.nsubmix_executed = hs(0);
  r.status.fallback_count = hs(1);
  r.status.material_negative_count = hs(2);
  r.status.input_negative_count = hs(3);
  auto hq = Kokkos::create_mirror_view(d.qcld);
  auto hr = Kokkos::create_mirror_view(d.raercol);
  auto hc = Kokkos::create_mirror_view(d.raercol_cw);
  Kokkos::deep_copy(hq, d.qcld);
  Kokkos::deep_copy(hr, d.raercol);
  Kokkos::deep_copy(hc, d.raercol_cw);
  r.state.N.assign(pver, 0.0);
  r.state.C.assign(pver, std::vector<double>(ncnst, 0.0));
  r.state.I.assign(pver, std::vector<double>(ncnst, 0.0));
  for (int k = 0; k < pver; ++k) {
    r.state.N[k] = hq(k);
    for (int mm = 0; mm < ncnst; ++mm) {
      r.state.C[k][mm] = hc(k, 0, mm);
      r.state.I[k][mm] = hr(k, 0, mm);
    }
  }
  return r;
}

bool bitwise_equal(const HostState &x, const HostState &y, const int top_lev) {
  for (int k = top_lev; k < pver; ++k) {
    if (x.N[k] != y.N[k])
      return false;
    for (int mm = 0; mm < ncnst; ++mm)
      if (x.C[k][mm] != y.C[k][mm] || x.I[k][mm] != y.I[k][mm])
        return false;
  }
  return true;
}

bool all_nonnegative(const HostState &x, const int top_lev) {
  for (int k = top_lev; k < pver; ++k) {
    if (x.N[k] < 0.0)
      return false;
    for (int mm = 0; mm < ncnst; ++mm)
      if (x.C[k][mm] < 0.0 || x.I[k][mm] < 0.0)
        return false;
  }
  return true;
}

double ledger_budget_rel() {
  const double eps = std::numeric_limits<Real>::epsilon();
  return std::max(5.0 * pver * eps, 1e-11);
}

// host replica of the acceptance budget: does zeroing the candidate's negative
// entries exceed the relative ledger budget for any field?
bool budget_violated(const HostOperator &op, const HostState &cand,
                     const HostState &old) {
  const int t = op.top_lev;
  for (int mm = -1; mm < ncnst; ++mm) {
    double L = 0.0, Z = 0.0;
    for (int k = t; k < pver; ++k) {
      if (mm < 0) {
        L += op.w[k] * old.N[k];
        Z += op.w[k] * std::max(0.0, -cand.N[k]);
      } else {
        L += op.w[k] * (old.C[k][mm] + old.I[k][mm]);
        Z += op.w[k] *
             (std::max(0.0, -cand.C[k][mm]) + std::max(0.0, -cand.I[k][mm]));
      }
    }
    if (!(Z <= ledger_budget_rel() * L))
      return true;
  }
  return false;
}

// least-squares slope and R^2 of log(err) vs log(h)
void fit_slope(const std::vector<double> &h, const std::vector<double> &err,
               double &slope, double &r2) {
  const int n = static_cast<int>(h.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  for (int i = 0; i < n; ++i) {
    const double x = std::log(h[i]), y = std::log(err[i]);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
  }
  slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const double num = n * sxy - sx * sy;
  r2 = (num * num) / ((n * sxx - sx * sx) * (n * syy - sy * sy));
}

} // namespace

// ===========================================================================
// 1. IMEX against the dense oracle, both policies, top_lev in {0, 6}
// ===========================================================================
TEST_CASE("imex_matches_dense_oracle", "mam4_ndrop_implicit_mixing") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop implicit mixing",
                                ekat::logger::LogLevel::info, comm);
  bool accepted_single_step = false;
  for (const std::string kind : {"stiff", "mild"})
    for (const int top_lev : {0, 6}) {
      for (const auto policy : {ImplicitSubstepPolicy::ControllerCount,
                                ImplicitSubstepPolicy::SingleStep}) {
        const HostCase c = make_case(kind, top_lev);
        const HostOperator op = make_operator(c);
        const int N = op.nsubmix(c.dtmicro);
        REQUIRE(N > 1);
        const bool single = policy == ImplicitSubstepPolicy::SingleStep;
        const int nsteps = single ? 1 : N;
        const double h = c.dtmicro / nsteps;
        // host oracle sequence; a step the budget would reject ends the
        // comparison (the production path then falls back to FE)
        HostState ref = host_state(c);
        bool predicted_reject = false;
        for (int i = 0; i < nsteps && !predicted_reject; ++i) {
          const HostState next = imex_oracle_step(op, ref, h);
          predicted_reject = budget_violated(op, next, ref);
          if (!predicted_reject)
            ref = next;
        }

        DeviceCase d = to_device(c);
        auto ws = ImplicitWorkspace::allocate("ws_imex");
        const auto r =
            run_device(d, top_lev, c.dtmicro, 1,
                       MixingOptions(MixingIntegrator::IMEX, policy), ws);
        logger.info(
            "IMEX top_lev={} single={} nsubmix_executed={} fallbacks={} "
            "predicted_reject={} ledger_rel={:.3e}",
            top_lev, single, r.status.nsubmix_executed, r.status.fallback_count,
            predicted_reject,
            max_relative_ledger_change(op, r.state, host_state(c)));
        REQUIRE(r.status.input_negative_count == 0);
        REQUIRE(max_relative_ledger_change(op, r.state, host_state(c)) <=
                ledger_budget_rel());
        REQUIRE(all_nonnegative(r.state, top_lev));
        if (!predicted_reject) {
          REQUIRE(r.status.fallback_count == 0);
          REQUIRE(r.status.nsubmix_executed == nsteps);
          REQUIRE(max_normalized_diff(op, r.state, ref) <= 1e-12);
          if (single)
            accepted_single_step = true;
        } else {
          // single step rejected: the full FE controller loop ran instead
          REQUIRE(single);
          REQUIRE(r.status.fallback_count == 1);
          REQUIRE(r.status.nsubmix_executed == N);
          DeviceCase dfe = to_device(c);
          const auto fe = run_device(dfe, top_lev, c.dtmicro, 1,
                                     MixingOptions(), ImplicitWorkspace());
          REQUIRE(bitwise_equal(r.state, fe.state, top_lev));
        }
      }
    }
  // the accepted single-step branch must be exercised by at least one column
  REQUIRE(accepted_single_step);
}

// ===========================================================================
// 2. Backward Euler against the 2n-by-2n dense oracle, both policies
// ===========================================================================
TEST_CASE("be_matches_dense_oracle", "mam4_ndrop_implicit_mixing") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop implicit mixing",
                                ekat::logger::LogLevel::info, comm);
  for (const int top_lev : {0, 6}) {
    for (const auto policy : {ImplicitSubstepPolicy::ControllerCount,
                              ImplicitSubstepPolicy::SingleStep}) {
      const HostCase c = make_case("stiff", top_lev);
      const HostOperator op = make_operator(c);
      const int N = op.nsubmix(c.dtmicro);
      const bool single = policy == ImplicitSubstepPolicy::SingleStep;
      const int nsteps = single ? 1 : N;
      const double h = c.dtmicro / nsteps;
      HostState ref = host_state(c);
      for (int i = 0; i < nsteps; ++i)
        ref = be_oracle_step(op, ref, h);

      DeviceCase d = to_device(c);
      auto ws = ImplicitWorkspace::allocate("ws_be");
      const auto r = run_device(
          d, top_lev, c.dtmicro, 1,
          MixingOptions(MixingIntegrator::BackwardEuler, policy), ws);
      logger.info("BE top_lev={} single={} nsubmix_executed={} fallbacks={} "
                  "max_norm_diff={:.3e} ledger_rel={:.3e}",
                  top_lev, single, r.status.nsubmix_executed,
                  r.status.fallback_count,
                  max_normalized_diff(op, r.state, ref),
                  max_relative_ledger_change(op, r.state, host_state(c)));
      REQUIRE(r.status.fallback_count == 0);
      REQUIRE(r.status.material_negative_count == 0);
      REQUIRE(r.status.nsubmix_executed == nsteps);
      REQUIRE(max_normalized_diff(op, r.state, ref) <= 1e-12);
      (void)N;
      REQUIRE(max_relative_ledger_change(op, r.state, host_state(c)) <=
              ledger_budget_rel());
      REQUIRE(all_nonnegative(r.state, top_lev));
    }
  }
}

// ===========================================================================
// 3. Fallback trigger: IMEX material negative on the two-level column
// ===========================================================================
TEST_CASE("imex_fallback_trigger_and_status", "mam4_ndrop_implicit_mixing") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop implicit mixing",
                                ekat::logger::LogLevel::info, comm);
  const HostCase c = make_case("two_level", pver - 2);
  const HostOperator op = make_operator(c);
  const int N = op.nsubmix(c.dtmicro);

  // plain Forward Euler reference (default options)
  DeviceCase dfe = to_device(c);
  const auto fe = run_device(dfe, c.top_lev, c.dtmicro, 1, MixingOptions(),
                             ImplicitWorkspace());
  REQUIRE(fe.status.nsubmix_executed == N);

  // SingleStep: the implicit step must be rejected by the conservation budget
  // and the full FE loop run, bitwise equal to plain FE
  {
    DeviceCase d = to_device(c);
    auto ws = ImplicitWorkspace::allocate("ws_fb_single");
    const auto r = run_device(d, c.top_lev, c.dtmicro, 1,
                              MixingOptions(MixingIntegrator::IMEX,
                                            ImplicitSubstepPolicy::SingleStep),
                              ws);
    logger.info(
        "IMEX single on two-level: fallbacks={} material={} executed={}",
        r.status.fallback_count, r.status.material_negative_count,
        r.status.nsubmix_executed);
    REQUIRE(r.status.fallback_count == 1);
    REQUIRE(r.status.material_negative_count == 1);
    REQUIRE(r.status.nsubmix_executed == N);
    REQUIRE(bitwise_equal(r.state, fe.state, c.top_lev));
  }
  // ControllerCount: at least the first substep is rejected and redone with
  // FE; the result is nonnegative and conservative within budget
  {
    DeviceCase d = to_device(c);
    auto ws = ImplicitWorkspace::allocate("ws_fb_ctrl");
    const auto r =
        run_device(d, c.top_lev, c.dtmicro, 1,
                   MixingOptions(MixingIntegrator::IMEX,
                                 ImplicitSubstepPolicy::ControllerCount),
                   ws);
    logger.info(
        "IMEX controller on two-level: fallbacks={} material={} executed={}",
        r.status.fallback_count, r.status.material_negative_count,
        r.status.nsubmix_executed);
    REQUIRE(r.status.fallback_count >= 1);
    REQUIRE(r.status.nsubmix_executed == N);
    REQUIRE(all_nonnegative(r.state, c.top_lev));
    REQUIRE(max_relative_ledger_change(op, r.state, host_state(c)) <=
            ledger_budget_rel());
    if (r.status.fallback_count == N)
      REQUIRE(bitwise_equal(r.state, fe.state, c.top_lev));
  }
  // Backward Euler is unconditionally positive on the same column
  {
    DeviceCase d = to_device(c);
    auto ws = ImplicitWorkspace::allocate("ws_fb_be");
    const auto r = run_device(d, c.top_lev, c.dtmicro, 1,
                              MixingOptions(MixingIntegrator::BackwardEuler,
                                            ImplicitSubstepPolicy::SingleStep),
                              ws);
    REQUIRE(r.status.fallback_count == 0);
    REQUIRE(r.status.nsubmix_executed == 1);
    REQUIRE(all_nonnegative(r.state, c.top_lev));
  }
}

// ===========================================================================
// 4. Invalid-state detection: input finiteness, candidate finiteness, pivot
// guard. The production path turns the input detector into an unrecoverable
// kernel error (Kokkos::abort), which this harness cannot catch, so the
// detectors are tested directly. The solver-status rejection cannot be driven
// through update_from_explmix with finite inputs (the pivot bound det >= 1
// holds), and a nonfinite coefficient would also poison the Forward Euler
// redo into the abort path; the guard is therefore exercised at detector level.
// ===========================================================================
TEST_CASE("invalid_state_detection", "mam4_ndrop_implicit_mixing") {
  using namespace mam4::ndrop::implicit_mixing;
  const HostCase c = make_case("stiff", 0);
  DeviceCase d = to_device(c);
  auto ws = ImplicitWorkspace::allocate("ws_invalid");
  mam4::DeviceType::view_1d<int> out("out", 8);
  Kokkos::deep_copy(out, 0);
  auto raercol = d.raercol, raercol_cw = d.raercol_cw;
  auto qcld = d.qcld, zn = d.zn, eddy_diff_kp = d.eddy_diff_kp,
       eddy_diff_km = d.eddy_diff_km, overlapp = d.overlapp,
       overlapm = d.overlapm;
  auto nact = d.nact, mact = d.mact;
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real inf = std::numeric_limits<Real>::infinity();
  auto policy = mam4::ThreadTeamPolicy(1u, team_size);
  Kokkos::parallel_for(
      "invalid_state_detection", policy,
      KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
        // clean input: no nonfinite entries
        const int clean =
            count_nonfinite_input(team, 0, 0, qcld, raercol, raercol_cw);
        // inject NaN and +Inf into the input state
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          raercol(10, 0, 3) = nan;
          raercol_cw(20, 0, 5) = inf;
        });
        team.team_barrier();
        const int dirty =
            count_nonfinite_input(team, 0, 0, qcld, raercol, raercol_cw);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          raercol(10, 0, 3) = Real(1);
          raercol_cw(20, 0, 5) = Real(1);
        });
        team.team_barrier();
        // candidate with a NaN entry is rejected by the finiteness check
        Kokkos::parallel_for(Kokkos::TeamVectorRange(team, pver), [&](int k) {
          for (int j = 0; j < ncnst + 1; ++j)
            ws.rhs_cloud(k, j) = Real(1);
          for (int mm = 0; mm < ncnst; ++mm)
            ws.rhs_inter(k, mm) = Real(1);
          eddy_diff_kp(k) = Real(1e-3);
          eddy_diff_km(k) = Real(1e-3);
          overlapp(k) = Real(1);
          overlapm(k) = Real(1);
        });
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { ws.rhs_inter(7, 2) = nan; });
        team.team_barrier();
        int nonfinite = 0, violations = 0;
        evaluate_acceptance(team, 0, 0, zn, qcld, raercol, raercol_cw, ws,
                            nonfinite, violations);
        // a candidate with a material negative is rejected by the budget
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          ws.rhs_inter(7, 2) = Real(1);
          ws.rhs_inter(8, 2) = Real(-1);
        });
        team.team_barrier();
        int nonfinite2 = 0, violations2 = 0;
        evaluate_acceptance(team, 0, 0, zn, qcld, raercol, raercol_cw, ws,
                            nonfinite2, violations2);
        // pivot guard: singular and nonfinite pivots are refused
        Mat2 inv;
        const bool ok_singular =
            invert_pivot(Mat2{Real(1), Real(2), Real(2), Real(4)}, inv);
        const bool ok_nan =
            invert_pivot(Mat2{nan, Real(0), Real(0), Real(1)}, inv);
        const bool ok_good =
            invert_pivot(Mat2{Real(2), Real(-0.5), Real(-0.5), Real(2)}, inv);
        // a family solve with a nonfinite coefficient fails the guard
        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { eddy_diff_kp(30) = nan; });
        team.team_barrier();
        const bool family_ok =
            be_family_solve(0, Real(100), 0, 0, nact, mact, raercol, raercol_cw,
                            eddy_diff_kp, eddy_diff_km, overlapp, overlapm, ws);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          out(0) = clean;
          out(1) = dirty;
          out(2) = nonfinite;
          out(3) = violations2;
          out(4) = ok_singular ? 1 : 0;
          out(5) = ok_nan ? 1 : 0;
          out(6) = ok_good ? 1 : 0;
          out(7) = family_ok ? 1 : 0;
        });
      });
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h, out);
  REQUIRE(h(0) == 0);
  REQUIRE(h(1) == 2);
  REQUIRE(h(2) >= 1);
  REQUIRE(h(3) >= 1);
  REQUIRE(h(4) == 0);
  REQUIRE(h(5) == 0);
  REQUIRE(h(6) == 1);
  REQUIRE(h(7) == 0);
}

// ===========================================================================
// 5. Temporal-accuracy gate on frozen operators: converged RK4 reference,
// first-order convergence of IMEX and BE, single-step errors recorded
// ===========================================================================
TEST_CASE("temporal_accuracy_gate", "mam4_ndrop_implicit_mixing") {
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop implicit mixing",
                                ekat::logger::LogLevel::info, comm);
  for (const std::string kind : {"stiff", "limiter"}) {
    const HostCase c = make_case(kind, 0);
    const HostOperator op = make_operator(c);
    const int N = op.nsubmix(c.dtmicro);
    REQUIRE(c.dtmicro / op.dtmin >= 50.0); // design: a genuinely stiff interval

    // converged reference: RK4 at dtmin/200, verified by step halving
    const int nref = static_cast<int>(std::ceil(200.0 * c.dtmicro / op.dtmin));
    const HostState ref = rk4_reference(op, host_state(c), c.dtmicro, nref);
    const HostState ref2 =
        rk4_reference(op, host_state(c), c.dtmicro, 2 * nref);
    const double ref_conv = max_normalized_diff(op, ref, ref2);
    logger.info("{}: N={} dtmin={:.2f} reference step-halving diff={:.3e}",
                kind, N, op.dtmin, ref_conv);
    REQUIRE(ref_conv <= 1e-9);

    // production FE at the controller count
    DeviceCase dfe = to_device(c);
    const auto fe =
        run_device(dfe, 0, c.dtmicro, 1, MixingOptions(), ImplicitWorkspace());
    const double err_fe = max_normalized_diff(op, fe.state, ref);
    logger.info("{}: FE controller N={} error={:.3e}", kind,
                fe.status.nsubmix_executed, err_fe);

    for (const auto integ :
         {MixingIntegrator::IMEX, MixingIntegrator::BackwardEuler}) {
      const char *name = integ == MixingIntegrator::IMEX ? "IMEX" : "BE";
      // controller-count row (one call, N substeps)
      {
        DeviceCase d = to_device(c);
        auto ws = ImplicitWorkspace::allocate("ws_acc_ctrl");
        const auto r = run_device(
            d, 0, c.dtmicro, 1,
            MixingOptions(integ, ImplicitSubstepPolicy::ControllerCount), ws);
        REQUIRE(r.status.fallback_count == 0);
        REQUIRE(r.status.nsubmix_executed == N);
        logger.info("{}: {} controller N={} error={:.3e}", kind, name, N,
                    max_normalized_diff(op, r.state, ref));
      }
      // convergence rows: N, 2N, 4N implicit steps of exactly h = T/(mN),
      // realized as mN single-step calls over T/(mN) each
      std::vector<double> hs, errs;
      for (const int mult : {1, 2, 4}) {
        const int nsplit = mult * N;
        DeviceCase d = to_device(c);
        auto ws = ImplicitWorkspace::allocate("ws_acc");
        const auto r = run_device(
            d, 0, c.dtmicro, nsplit,
            MixingOptions(integ, ImplicitSubstepPolicy::SingleStep), ws);
        REQUIRE(r.status.fallback_count == 0);
        REQUIRE(r.status.nsubmix_executed == nsplit);
        const double h = c.dtmicro / nsplit;
        const double err = max_normalized_diff(op, r.state, ref);
        hs.push_back(h);
        errs.push_back(err);
        logger.info("{}: {} steps={} h={:.3f} error={:.3e}", kind, name, nsplit,
                    h, err);
      }
      double slope, r2;
      fit_slope(hs, errs, slope, r2);
      logger.info("{}: {} convergence slope={:.3f} R2={:.4f}", kind, name,
                  slope, r2);
      REQUIRE(slope >= 0.9);
      REQUIRE(r2 >= 0.99);

      DeviceCase d1 = to_device(c);
      auto ws1 = ImplicitWorkspace::allocate("ws_acc_single");
      const auto single = run_device(
          d1, 0, c.dtmicro, 1,
          MixingOptions(integ, ImplicitSubstepPolicy::SingleStep), ws1);
      const double err_single = max_normalized_diff(op, single.state, ref);
      logger.info(
          "{}: {} single-step error={:.3e} (FE controller error {:.3e}, "
          "ratio {:.2f}, fallbacks={})",
          kind, name, err_single, err_fe, err_single / err_fe,
          single.status.fallback_count);
      REQUIRE(std::isfinite(err_single));
      // interim accuracy bound (open decision 5): report only
      if (err_single > 10.0 * err_fe)
        logger.warn("{}: {} single step exceeds the interim bound of ten times "
                    "the FE controller error; accuracy-unqualified",
                    kind, name);
    }
  }
}

// ===========================================================================
// 6. Residual verification across the supported coefficient range: one
// substep of each implicit method on directly specified coefficients, residual
// of the discrete equations evaluated on the host against the device solution
// ===========================================================================
namespace {

struct GridArrays {
  mam4::ColumnView zn, qcld, eddy_diff_kp, eddy_diff_km, overlapp, overlapm;
  mam4::ndrop::View2D nact, mact;
  mam4::ndrop::View3D raercol, raercol_cw;
};

// uniform per-direction rate r so h (a_k + b_k) = htinv in the interior
GridArrays make_grid_arrays(const double r, const double overlap,
                            const double act_ratio, HostOperator &op,
                            HostState &s0) {
  GridArrays g;
  g.zn = mam4::testing::create_column_view(pver);
  g.qcld = mam4::testing::create_column_view(pver);
  g.eddy_diff_kp = mam4::testing::create_column_view(pver);
  g.eddy_diff_km = mam4::testing::create_column_view(pver);
  g.overlapp = mam4::testing::create_column_view(pver);
  g.overlapm = mam4::testing::create_column_view(pver);
  g.nact = mam4::ndrop::View2D("g_nact", pver, nmodes);
  g.mact = mam4::ndrop::View2D("g_mact", pver, nmodes);
  g.raercol = mam4::ndrop::View3D("g_raercol", pver, 2, ncnst);
  g.raercol_cw = mam4::ndrop::View3D("g_raercol_cw", pver, 2, ncnst);
  auto hzn = Kokkos::create_mirror_view(g.zn);
  auto hq = Kokkos::create_mirror_view(g.qcld);
  auto hkp = Kokkos::create_mirror_view(g.eddy_diff_kp);
  auto hkm = Kokkos::create_mirror_view(g.eddy_diff_km);
  auto hop = Kokkos::create_mirror_view(g.overlapp);
  auto hom = Kokkos::create_mirror_view(g.overlapm);
  auto hn = Kokkos::create_mirror_view(g.nact);
  auto hm = Kokkos::create_mirror_view(g.mact);
  auto hr = Kokkos::create_mirror_view(g.raercol);
  auto hc = Kokkos::create_mirror_view(g.raercol_cw);
  op.top_lev = 0;
  op.a.assign(pver, 0.0);
  op.b.assign(pver, 0.0);
  op.oA.assign(pver, overlap);
  op.oB.assign(pver, overlap);
  op.w.assign(pver, 1.0);
  op.act_n.assign(pver, std::vector<double>(nmodes, 0.0));
  op.act_m.assign(pver, std::vector<double>(nmodes, 0.0));
  s0.N.assign(pver, 0.0);
  s0.C.assign(pver, std::vector<double>(ncnst, 0.0));
  s0.I.assign(pver, std::vector<double>(ncnst, 0.0));
  for (int k = 0; k < pver; ++k) {
    const double b = (k == pver - 1) ? 0.0 : r;
    const double a = (k == 0) ? 0.0 : r;
    const double act = act_ratio * b;
    hzn(k) = Real(1);
    hkp(k) = static_cast<Real>(b);
    hkm(k) = static_cast<Real>(a);
    hop(k) = static_cast<Real>(overlap);
    hom(k) = static_cast<Real>(overlap);
    op.a[k] = a;
    op.b[k] = b;
    const double prof = 1.0 + 0.5 * double(k) / double(pver - 1);
    s0.N[k] = 1.0e7 * prof;
    hq(k) = static_cast<Real>(s0.N[k]);
    for (int m = 0; m < nmodes; ++m) {
      hn(k, m) = static_cast<Real>(act);
      hm(k, m) = static_cast<Real>(act);
      op.act_n[k][m] = act;
      op.act_m[k][m] = act;
    }
    for (int mm = 0; mm < ncnst; ++mm) {
      s0.I[k][mm] = 1.0e9 * prof * (1.0 + 0.01 * mm);
      s0.C[k][mm] = 0.3e9 * prof;
      hr(k, 0, mm) = static_cast<Real>(s0.I[k][mm]);
      hc(k, 0, mm) = static_cast<Real>(s0.C[k][mm]);
      hr(k, 1, mm) = Real(-7);
      hc(k, 1, mm) = Real(-7);
    }
  }
  Kokkos::deep_copy(g.zn, hzn);
  Kokkos::deep_copy(g.qcld, hq);
  Kokkos::deep_copy(g.eddy_diff_kp, hkp);
  Kokkos::deep_copy(g.eddy_diff_km, hkm);
  Kokkos::deep_copy(g.overlapp, hop);
  Kokkos::deep_copy(g.overlapm, hom);
  Kokkos::deep_copy(g.nact, hn);
  Kokkos::deep_copy(g.mact, hm);
  Kokkos::deep_copy(g.raercol, hr);
  Kokkos::deep_copy(g.raercol_cw, hc);
  return g;
}

// copy the candidate held in a workspace to a HostState
HostState candidate_to_host(const ImplicitWorkspace &ws) {
  HostState s;
  s.N.assign(pver, 0.0);
  s.C.assign(pver, std::vector<double>(ncnst, 0.0));
  s.I.assign(pver, std::vector<double>(ncnst, 0.0));
  auto hc = Kokkos::create_mirror_view(ws.rhs_cloud);
  auto hi = Kokkos::create_mirror_view(ws.rhs_inter);
  Kokkos::deep_copy(hc, ws.rhs_cloud);
  Kokkos::deep_copy(hi, ws.rhs_inter);
  for (int k = 0; k < pver; ++k) {
    s.N[k] = hc(k, 0);
    for (int mm = 0; mm < ncnst; ++mm) {
      s.C[k][mm] = hc(k, 1 + mm);
      s.I[k][mm] = hi(k, mm);
    }
  }
  return s;
}

double norm_inf(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v)
    m = std::max(m, std::fabs(x));
  return m;
}
double matrix_norm_inf(const Mat &A) {
  double m = 0.0;
  for (const auto &row : A) {
    double r = 0.0;
    for (double x : row)
      r += std::fabs(x);
    m = std::max(m, r);
  }
  return m;
}
// ||A x - b||_inf / (||A||_inf ||x||_inf + ||b||_inf)
double normalized_residual(const Mat &A, const std::vector<double> &x,
                           const std::vector<double> &b) {
  const int n = static_cast<int>(b.size());
  std::vector<double> res(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double s = -b[i];
    for (int j = 0; j < n; ++j)
      s += A[i][j] * x[j];
    res[i] = s;
  }
  return norm_inf(res) / (matrix_norm_inf(A) * norm_inf(x) + norm_inf(b));
}

// BE 2n-by-2n system of constituent mm (same equations as be_oracle_step)
void be_system(const HostOperator &op, const HostState &s, const double h,
               const int mm, Mat &A, std::vector<double> &rhs) {
  const int t = op.top_lev, n = pver - t;
  A.assign(2 * n, std::vector<double>(2 * n, 0.0));
  rhs.assign(2 * n, 0.0);
  for (int k = t; k < pver; ++k) {
    const int r = k - t;
    const int kp1 = std::min(k + 1, pver - 1);
    const int km1 = std::max(k - 1, t);
    const double act = act_of(op, k, mm);
    const int iC = 2 * r, iI = 2 * r + 1;
    A[iC][iC] += 1.0 + h * (op.a[k] + op.b[k]);
    if (k > t)
      A[iC][2 * (km1 - t)] += -h * op.a[k] * op.oA[k];
    if (k < pver - 1) {
      A[iC][2 * (kp1 - t)] += -h * op.b[k] * op.oB[k];
      A[iC][2 * (kp1 - t) + 1] += -h * act;
    }
    rhs[iC] = s.C[k][mm];
    A[iI][iI] += 1.0 + h * (op.a[k] + op.b[k]);
    if (k > t) {
      A[iI][2 * (km1 - t) + 1] += -h * op.a[k];
      A[iI][2 * (km1 - t)] += -h * op.a[k] * (1.0 - op.oA[k]);
    }
    if (k < pver - 1) {
      A[iI][2 * (kp1 - t) + 1] += -h * op.b[k] + h * act;
      A[iI][2 * (kp1 - t)] += -h * op.b[k] * (1.0 - op.oB[k]);
    }
    rhs[iI] = s.I[k][mm];
  }
}

} // namespace

TEST_CASE("residual_verification_across_coefficient_range",
          "mam4_ndrop_implicit_mixing") {
  using namespace mam4::ndrop::implicit_mixing;
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop implicit mixing",
                                ekat::logger::LogLevel::info, comm);
  const double h = 1800.0;
  const double tol = 100.0 * pver * std::numeric_limits<Real>::epsilon();
  double worst_imex = 0.0, worst_be = 0.0;
  for (const double htinv : {1.0, 10.0, 100.0, 450.0}) {
    for (const double overlap : {0.0, 0.5, 1.0}) {
      for (const double act_ratio : {0.0, 0.5, 1.0}) {
        HostOperator op;
        HostState s0;
        const double r = htinv / (2.0 * h);
        GridArrays g = make_grid_arrays(r, overlap, act_ratio, op, s0);
        auto ws_imex = ImplicitWorkspace::allocate("ws_grid_imex");
        auto ws_be = ImplicitWorkspace::allocate("ws_grid_be");
        mam4::DeviceType::view_1d<int> be_ok("be_ok", 1);
        auto zn = g.zn, qcld = g.qcld, kp = g.eddy_diff_kp, km = g.eddy_diff_km,
             op_ = g.overlapp, om = g.overlapm;
        auto nact = g.nact, mact = g.mact;
        auto raercol = g.raercol, raercol_cw = g.raercol_cw;
        const Real hr = static_cast<Real>(h);
        auto policy = mam4::ThreadTeamPolicy(1u, team_size);
        Kokkos::parallel_for(
            "residual_grid", policy,
            KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
              imex_substep(team, hr, 0, 0, nact, mact, qcld, raercol,
                           raercol_cw, kp, km, op_, om, ws_imex);
              const bool ok =
                  be_substep(team, hr, 0, 0, nact, mact, qcld, raercol,
                             raercol_cw, kp, km, op_, om, ws_be);
              Kokkos::single(Kokkos::PerTeam(team),
                             [&]() { be_ok(0) = ok ? 1 : 0; });
            });
        Kokkos::fence();
        auto hok = Kokkos::create_mirror_view(be_ok);
        Kokkos::deep_copy(hok, be_ok);
        REQUIRE(hok(0) == 1);
        const HostState xi = candidate_to_host(ws_imex);
        const HostState xb = candidate_to_host(ws_be);

        // IMEX residuals: A_C C = C^n + h S(I^n); A_I I = I^n - h S +
        // return(C^{n+1})
        const Mat AC = scalar_matrix(op, h, true);
        const Mat AI = scalar_matrix(op, h, false);
        double res_imex = 0.0;
        {
          std::vector<double> x(pver), b(pver);
          for (int k = 0; k < pver; ++k) {
            const int kp1 = std::min(k + 1, pver - 1);
            double srcn = 0.0;
            for (int m = 0; m < ::nmodes; ++m)
              srcn +=
                  op.act_n[k][m] * s0.I[kp1][mam4::AeroConfig::mam_idx(m, 0)];
            x[k] = xi.N[k];
            b[k] = s0.N[k] + h * srcn;
          }
          res_imex = std::max(res_imex, normalized_residual(AC, x, b));
          for (int mm = 0; mm < ncnst; ++mm) {
            for (int k = 0; k < pver; ++k) {
              const int kp1 = std::min(k + 1, pver - 1);
              x[k] = xi.C[k][mm];
              b[k] = s0.C[k][mm] + h * act_of(op, k, mm) * s0.I[kp1][mm];
            }
            res_imex = std::max(res_imex, normalized_residual(AC, x, b));
            for (int k = 0; k < pver; ++k) {
              const int kp1 = std::min(k + 1, pver - 1);
              const int km1 = std::max(k - 1, 0);
              x[k] = xi.I[k][mm];
              b[k] = s0.I[k][mm] - h * act_of(op, k, mm) * s0.I[kp1][mm] +
                     h * op.a[k] * (1.0 - op.oA[k]) * xi.C[km1][mm] +
                     h * op.b[k] * (1.0 - op.oB[k]) * xi.C[kp1][mm];
            }
            res_imex = std::max(res_imex, normalized_residual(AI, x, b));
          }
        }
        // BE residuals: 2n-by-2n per constituent, then droplets
        double res_be = 0.0;
        {
          Mat A;
          std::vector<double> b, x(2 * pver);
          for (int mm = 0; mm < ncnst; ++mm) {
            be_system(op, s0, h, mm, A, b);
            for (int k = 0; k < pver; ++k) {
              x[2 * k] = xb.C[k][mm];
              x[2 * k + 1] = xb.I[k][mm];
            }
            res_be = std::max(res_be, normalized_residual(A, x, b));
          }
          std::vector<double> xn(pver), bn(pver);
          for (int k = 0; k < pver; ++k) {
            const int kp1 = std::min(k + 1, pver - 1);
            double srcn = 0.0;
            for (int m = 0; m < ::nmodes; ++m)
              srcn +=
                  op.act_n[k][m] * xb.I[kp1][mam4::AeroConfig::mam_idx(m, 0)];
            xn[k] = xb.N[k];
            bn[k] = s0.N[k] + h * srcn;
          }
          res_be = std::max(res_be, normalized_residual(AC, xn, bn));
        }
        worst_imex = std::max(worst_imex, res_imex);
        worst_be = std::max(worst_be, res_be);
        REQUIRE(res_imex <= tol);
        REQUIRE(res_be <= tol);
      }
    }
  }
  logger.info("residual grid (36 points, h tinv up to 450): worst IMEX {:.3e}, "
              "worst BE {:.3e}, tolerance {:.3e}",
              worst_imex, worst_be, tol);
}

// ===========================================================================
// 7. Two-stage BE kernel against a serial reference performing the same two
// stages (a race detector for GPU builds; trivially equal with one thread)
// ===========================================================================
TEST_CASE("be_two_stage_kernel_matches_serial", "mam4_ndrop_implicit_mixing") {
  using namespace mam4::ndrop::implicit_mixing;
  HostOperator op;
  HostState s0;
  GridArrays g = make_grid_arrays(100.0 / (2.0 * 1800.0), 0.5, 0.5, op, s0);
  auto ws_par = ImplicitWorkspace::allocate("ws_par");
  auto ws_ser = ImplicitWorkspace::allocate("ws_ser");
  auto qcld = g.qcld, kp = g.eddy_diff_kp, km = g.eddy_diff_km,
       op_ = g.overlapp, om = g.overlapm;
  auto nact = g.nact, mact = g.mact;
  auto raercol = g.raercol, raercol_cw = g.raercol_cw;
  const Real hr = Real(1800);
  auto policy = mam4::ThreadTeamPolicy(1u, team_size);
  Kokkos::parallel_for(
      "be_two_stage", policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
        // stage 1 + stage 2 as implemented
        be_substep(team, hr, 0, 0, nact, mact, qcld, raercol, raercol_cw, kp,
                   km, op_, om, ws_par);
        // serial reference: families one after another on one thread, then
        // the same droplet stage
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          for (int f = 0; f < nfamilies; ++f)
            be_family_solve(f, hr, 0, 0, nact, mact, raercol, raercol_cw, kp,
                            km, op_, om, ws_ser);
        });
        team.team_barrier();
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, 0, pver), [&](int k) {
              const int kp1 = mam4::min(k + 1, pver - 1);
              Real srcn = Real(0);
              for (int imode = 0; imode < ::nmodes; ++imode)
                srcn +=
                    nact(k, imode) *
                    ws_ser.rhs_inter(kp1, mam4::AeroConfig::mam_idx(imode, 0));
              ws_ser.rhs_drop(k, 0) = qcld(k) + hr * srcn;
            });
        team.team_barrier();
        assemble_bands(team, 0, hr, true, kp, km, op_, om, ws_ser.dl, ws_ser.d,
                       ws_ser.du);
        solve_tridiagonal(team, ws_ser.dl, ws_ser.d, ws_ser.du,
                          ws_ser.rhs_drop);
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, 0, pver),
            [&](int k) { ws_ser.rhs_cloud(k, 0) = ws_ser.rhs_drop(k, 0); });
      });
  Kokkos::fence();
  const HostState a = candidate_to_host(ws_par);
  const HostState b = candidate_to_host(ws_ser);
  REQUIRE(bitwise_equal(a, b, 0));
}

// ===========================================================================
// 8. Every EKAT tridiagonal backend that solve_tridiagonal can dispatch to
// (bfb, cyclic reduction, serial Thomas) is driven directly on the production
// bands, so the branches this build does not select still compile and solve
// the same system to oracle tolerance. Deliberately not a bitwise check: the
// backends order the arithmetic differently, which is why implicit results
// are documented as not BFB across CPU, GPU, and Debug builds.
// ===========================================================================
TEST_CASE("tridiagonal_backends_agree_with_oracle",
          "mam4_ndrop_implicit_mixing") {
  using namespace mam4::ndrop::implicit_mixing;
  using mam4::ndrop::View1D;
  using mam4::ndrop::View2D;
  ekat::Comm comm;
  ekat::logger::Logger<> logger("ndrop implicit mixing",
                                ekat::logger::LogLevel::info, comm);
  const double h = 1800.0;
  const double tol = 100.0 * pver * std::numeric_limits<Real>::epsilon();
  const char *names[3] = {"bfb", "cr", "thomas"};
  double worst_res = 0.0, worst_spread = 0.0;
  for (const double htinv : {10.0, 450.0}) {
    for (const bool cloud_family : {true, false}) {
      HostOperator op;
      HostState s0;
      GridArrays g = make_grid_arrays(htinv / (2.0 * h), 0.5, 0.5, op, s0);
      View1D dl("bk_dl", pver), d("bk_d", pver), du("bk_du", pver);
      View1D dl0("bk_dl0", pver), d0("bk_d0", pver), du0("bk_du0", pver);
      View1D dl1("bk_dl1", pver), d1("bk_d1", pver), du1("bk_du1", pver);
      View1D dl2("bk_dl2", pver), d2("bk_d2", pver), du2("bk_du2", pver);
      View2D X0("bk_X0", pver, ncnst), X1("bk_X1", pver, ncnst),
          X2("bk_X2", pver, ncnst);
      auto kp = g.eddy_diff_kp, km = g.eddy_diff_km, op_ = g.overlapp,
           om = g.overlapm;
      auto raercol = g.raercol;
      const Real hr = static_cast<Real>(h);
      auto policy = mam4::ThreadTeamPolicy(1u, team_size);
      Kokkos::parallel_for(
          "tridiag_backends", policy,
          KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
            assemble_bands(team, 0, hr, cloud_family, kp, km, op_, om, dl, d,
                           du);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, 0, pver), [&](int k) {
                  dl0(k) = dl1(k) = dl2(k) = dl(k);
                  d0(k) = d1(k) = d2(k) = d(k);
                  du0(k) = du1(k) = du2(k) = du(k);
                  for (int mm = 0; mm < ncnst; ++mm)
                    X0(k, mm) = X1(k, mm) = X2(k, mm) = raercol(k, 0, mm);
                });
            team.team_barrier();
            ekat::tridiag::bfb(team, dl0, d0, du0, X0);
            team.team_barrier();
            ekat::tridiag::cr(team, dl1, d1, du1, X1);
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team),
                           [&]() { ekat::tridiag::thomas(dl2, d2, du2, X2); });
            team.team_barrier();
          });
      Kokkos::fence();
      auto h0 = Kokkos::create_mirror_view(X0);
      auto h1 = Kokkos::create_mirror_view(X1);
      auto h2 = Kokkos::create_mirror_view(X2);
      Kokkos::deep_copy(h0, X0);
      Kokkos::deep_copy(h1, X1);
      Kokkos::deep_copy(h2, X2);
      const Mat A = scalar_matrix(op, h, cloud_family);
      std::vector<double> x(pver), b(pver);
      for (int mm = 0; mm < ncnst; ++mm) {
        for (int k = 0; k < pver; ++k)
          b[k] = s0.I[k][mm];
        double xnorm = 0.0;
        for (int i = 0; i < 3; ++i) {
          for (int k = 0; k < pver; ++k)
            x[k] = (i == 0) ? h0(k, mm) : (i == 1) ? h1(k, mm) : h2(k, mm);
          const double res = normalized_residual(A, x, b);
          worst_res = std::max(worst_res, res);
          if (res > tol)
            logger.error("backend {} residual {:.3e} > {:.3e} (h tinv {}, {})",
                         names[i], res, tol, htinv,
                         cloud_family ? "A_C" : "A_I");
          REQUIRE(res <= tol);
          xnorm = std::max(xnorm, norm_inf(x));
        }
        for (int k = 0; k < pver; ++k) {
          const double spread =
              std::max({std::fabs(double(h0(k, mm)) - double(h1(k, mm))),
                        std::fabs(double(h0(k, mm)) - double(h2(k, mm))),
                        std::fabs(double(h1(k, mm)) - double(h2(k, mm)))}) /
              xnorm;
          worst_spread = std::max(worst_spread, spread);
          REQUIRE(spread <= tol);
        }
      }
    }
  }
  logger.info("tridiagonal backends (bfb, cr, thomas) on A_C and A_I, h tinv "
              "10 and 450: worst residual {:.3e}, worst relative spread "
              "between backends {:.3e}, tolerance {:.3e}",
              worst_res, worst_spread, tol);
}
