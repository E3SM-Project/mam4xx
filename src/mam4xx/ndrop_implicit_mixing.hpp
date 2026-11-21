// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_NDROP_IMPLICIT_MIXING_HPP
#define MAM4XX_NDROP_IMPLICIT_MIXING_HPP

#ifndef MAM4XX_NDROP_HPP
#error "ndrop_implicit_mixing.hpp is included by ndrop.hpp; include that header"
#endif

// Optional implicit vertical-mixing integrators for ndrop::update_from_explmix.
//
// This header provides two runtime-selectable alternatives to the explicit
// (Forward Euler, FE) mixing substep loop:
//
//   * IMEX-Euler: activation explicit from the old state; vertical mixing and
//     the cloud-to-interstitial return implicit. Two shared scalar tridiagonal
//     families (cloud-borne + droplets; interstitial) solved with EKAT.
//   * Backward Euler (BE): mixing, return, and activation all implicit. One
//     2-by-2 block-tridiagonal family per (mode, number|mass), solved with a
//     guarded block-Thomas sweep, then a scalar droplet solve.
//
// Both use the production directional rates eddy_diff_kp (toward k+1) and
// eddy_diff_km (toward k-1), the production overlaps, and the capped
// activation rates exactly as update_from_explmix forms them. The first
// active level's eddy_diff_km term and the surface cell's eddy_diff_kp term
// are identically zero in the FE update (clamped neighbour, overlap 1), and
// are excluded from every matrix here.
//
// The default FE path in update_from_explmix is untouched. This header is
// included from ndrop.hpp after explmix() is defined and relies on the
// ndrop namespace types and constants; it is not meant to be included alone.

#include <ekat_kernel_assert.hpp>
#include <ekat_tridiag.hpp>

#include <limits>
#include <string>
#include <type_traits>

namespace mam4 {
namespace ndrop {

// EKAT's serial Thomas solver requires contiguous LayoutRight arrays; the
// workspace relies on ekat::KokkosTypes using that layout for view_2d.
static_assert(std::is_same<View2D::array_layout, Kokkos::LayoutRight>::value,
              "ndrop implicit mixing requires LayoutRight right-hand sides");

enum class MixingIntegrator : int {
  ForwardEuler = 0,
  IMEX = 1,
  BackwardEuler = 2
};

enum class ImplicitSubstepPolicy : int { ControllerCount = 0, SingleStep = 1 };

// Runtime options. The defaults reproduce the existing explicit path exactly.
struct MixingOptions {
  MixingIntegrator integrator;
  ImplicitSubstepPolicy policy;
  KOKKOS_INLINE_FUNCTION
  MixingOptions()
      : integrator(MixingIntegrator::ForwardEuler),
        policy(ImplicitSubstepPolicy::ControllerCount) {}
  KOKKOS_INLINE_FUNCTION
  MixingOptions(const MixingIntegrator i, const ImplicitSubstepPolicy p)
      : integrator(i), policy(p) {}
};

// Caller-provided per-column scratch for the implicit paths. All views are
// LayoutRight (ekat::KokkosTypes default), as the EKAT tridiagonal solvers
// require. Nothing is allocated inside kernels.
struct ImplicitWorkspace {
  // candidate cloud-borne state: column 0 droplets, column 1+mm constituent mm
  View2D rhs_cloud; // (pver, 1 + ncnst_tot)
  // candidate interstitial state: column mm constituent mm
  View2D rhs_inter; // (pver, ncnst_tot)
  // contiguous droplet right-hand side for the Backward Euler scalar solve
  View2D rhs_drop; // (pver, 1)
  // scalar tridiagonal bands (overwritten by every solve)
  View1D dl, d, du; // (pver)
  // Backward Euler eliminated 2-by-2 pivot inverses per level and family
  View3D be_factor; // (pver, 2 * num_modes, 4)

  KOKKOS_INLINE_FUNCTION ImplicitWorkspace() {}

  KOKKOS_INLINE_FUNCTION bool valid() const {
    constexpr int nfam = 2 * AeroConfig::num_modes();
    return rhs_cloud.span_is_contiguous() && rhs_inter.span_is_contiguous() &&
           rhs_drop.span_is_contiguous() && rhs_cloud.extent_int(0) == pver &&
           rhs_cloud.extent_int(1) == ncnst_tot + 1 &&
           rhs_inter.extent_int(0) == pver &&
           rhs_inter.extent_int(1) == ncnst_tot &&
           rhs_drop.extent_int(0) == pver && rhs_drop.extent_int(1) == 1 &&
           dl.extent_int(0) == pver && d.extent_int(0) == pver &&
           du.extent_int(0) == pver && be_factor.extent_int(0) == pver &&
           be_factor.extent_int(1) == nfam && be_factor.extent_int(2) == 4;
  }

  // Host-side allocation helper for one column.
  static ImplicitWorkspace allocate(const std::string &prefix) {
    constexpr int nfam = 2 * AeroConfig::num_modes();
    ImplicitWorkspace ws;
    ws.rhs_cloud = View2D(prefix + "_rhs_cloud", pver, ncnst_tot + 1);
    ws.rhs_inter = View2D(prefix + "_rhs_inter", pver, ncnst_tot);
    ws.rhs_drop = View2D(prefix + "_rhs_drop", pver, 1);
    ws.dl = View1D(prefix + "_dl", pver);
    ws.d = View1D(prefix + "_d", pver);
    ws.du = View1D(prefix + "_du", pver);
    ws.be_factor = View3D(prefix + "_be_factor", pver, nfam, 4);
    return ws;
  }
};

// Per-column execution record.
struct MixingStatus {
  // total substeps actually run for the column: accepted implicit substeps
  // plus any Forward Euler redo substeps
  int nsubmix_executed;
  // substeps rejected by the acceptance checks and redone with Forward Euler
  // (under SingleStep, one rejection runs the full controller loop)
  int fallback_count;
  // substeps rejected for any negative candidate, including roundoff
  int material_negative_count;
  // negative entries in the input state (host-supplied), both phases + qcld;
  // counted only when an implicit integrator runs (the explicit path reports 0)
  int input_negative_count;
  KOKKOS_INLINE_FUNCTION
  MixingStatus()
      : nsubmix_executed(0), fallback_count(0), material_negative_count(0),
        input_negative_count(0) {}
};

namespace implicit_mixing {

constexpr int nmodes = AeroConfig::num_modes();
constexpr int nfamilies = 2 * nmodes; // number families then mass families
constexpr int ncloud_rhs = ncnst_tot + 1;

KOKKOS_INLINE_FUNCTION Real machine_epsilon() {
  return std::numeric_limits<Real>::epsilon();
}
KOKKOS_INLINE_FUNCTION Real real_max() {
  return std::numeric_limits<Real>::max();
}
KOKKOS_INLINE_FUNCTION bool is_finite(const Real x) {
  // true for finite values only; false for NaN and both infinities. Like the
  // existing mam4xx finiteness guards, this is defeated by builds that enable
  // finite-math-only optimizations.
  return x <= real_max() && x >= -real_max();
}

// Directional rates with the clamped-neighbour terms removed. In the FE update
// the eddy_diff_km term at the first active level and the eddy_diff_kp term at
// the surface cell are identically zero (overlap 1, neighbour clamped to the
// level itself), so they must not enter any matrix.
KOKKOS_INLINE_FUNCTION Real rate_up(const int k, const int top_lev,
                                    const ColumnView &eddy_diff_km) {
  return k == top_lev ? Real(0) : eddy_diff_km(k);
}
KOKKOS_INLINE_FUNCTION Real rate_down(const int k,
                                      const ColumnView &eddy_diff_kp) {
  return k == pver - 1 ? Real(0) : eddy_diff_kp(k);
}

// ---------------------------------------------------------------------------
// Input inspection (once per call)
// ---------------------------------------------------------------------------

// Number of nonfinite entries in the old state (both phases and droplets).
KOKKOS_INLINE_FUNCTION
int count_nonfinite_input(const ThreadTeam &team, const int top_lev,
                          const int nsav, const ColumnView &qcld,
                          const View3D &raercol, const View3D &raercol_cw) {
  int count = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, top_lev, pver),
      [&](const int k, int &acc) {
        if (!is_finite(qcld(k)))
          ++acc;
        for (int mm = 0; mm < ncnst_tot; ++mm) {
          if (!is_finite(raercol(k, nsav, mm)))
            ++acc;
          if (!is_finite(raercol_cw(k, nsav, mm)))
            ++acc;
        }
      },
      count);
  team.team_barrier();
  return count;
}

// Number of negative entries in the old state (both phases and droplets).
KOKKOS_INLINE_FUNCTION
int count_negative_input(const ThreadTeam &team, const int top_lev,
                         const int nsav, const ColumnView &qcld,
                         const View3D &raercol, const View3D &raercol_cw) {
  int count = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, top_lev, pver),
      [&](const int k, int &acc) {
        if (qcld(k) < Real(0))
          ++acc;
        for (int mm = 0; mm < ncnst_tot; ++mm) {
          if (raercol(k, nsav, mm) < Real(0))
            ++acc;
          if (raercol_cw(k, nsav, mm) < Real(0))
            ++acc;
        }
      },
      count);
  team.team_barrier();
  return count;
}

// The verified coefficient range is enforced on physical rates only.
// Compare without overflowing h*rate for otherwise valid finite inputs.
KOKKOS_INLINE_FUNCTION
bool within_coefficient_range(const Real h, const Real rate) {
  return rate <= Real(1) ? h * rate <= Real(500) : h <= Real(500) / rate;
}
KOKKOS_INLINE_FUNCTION
bool within_implicit_range(const ThreadTeam &team, const int top_lev,
                           const Real h, const ColumnView &kp,
                           const ColumnView &km) {
  int outside = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, top_lev, pver),
      [&](int k, int &acc) {
        if (!within_coefficient_range(h, rate_up(k, top_lev, km)) ||
            !within_coefficient_range(h, rate_down(k, kp)))
          ++acc;
      },
      outside);
  team.team_barrier();
  return pver <= 128 && outside == 0;
}

// Structural preconditions: nonnegative capped activation rates everywhere
// and zero activation at the surface cell (always true in dropmixnuc).
KOKKOS_INLINE_FUNCTION
bool activation_preconditions_hold(const ThreadTeam &team, const int top_lev,
                                   const View2D &nact, const View2D &mact,
                                   const ColumnView &eddy_diff_kp) {
  int violations = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, top_lev, pver),
      [&](const int k, int &acc) {
        for (int imode = 0; imode < nmodes; ++imode) {
          const Real n = nact(k, imode);
          const Real m = mact(k, imode);
          if (!is_finite(n) || !is_finite(m) || n < Real(0) || m < Real(0) ||
              n > eddy_diff_kp(k) || m > eddy_diff_kp(k))
            ++acc;
          if (k == pver - 1 && (n != Real(0) || m != Real(0)))
            ++acc;
        }
      },
      violations);
  team.team_barrier();
  return violations == 0;
}

// ---------------------------------------------------------------------------
// Forward Euler substep (a separately factored copy of the loop body in
// update_from_explmix, used only by the fallback paths; the default path keeps
// the original loop)
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void forward_euler_substep(const ThreadTeam &team, const Real dtmix,
                           const int top_lev, const int nsav, const int nnew,
                           const View2D &nact, const View2D &mact,
                           const ColumnView &qcld, const ColumnView &qncld,
                           const View3D &raercol, const View3D &raercol_cw,
                           const ColumnView &eddy_diff_kp,
                           const ColumnView &eddy_diff_km,
                           const ColumnView &overlapp,
                           const ColumnView &overlapm) {
  const Real zero = 0.0;
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, top_lev, pver),
                       [&](int k) { qncld(k) = qcld(k); });
  team.team_barrier();
  Kokkos::parallel_for(
      Kokkos::TeamVectorRange(team, top_lev, pver), [&](int k) {
        const int kp1 = mam4::min(k + 1, pver - 1);
        const int km1 = mam4::max(k - 1, top_lev);
        Real srcn = zero;
        for (int imode = 0; imode < nmodes; imode++) {
          const int mm = AeroConfig::mam_idx(imode, 0);
          srcn += nact(k, imode) * raercol(kp1, nsav, mm);
          if (k == pver - 1) {
            const Real tmpa = raercol(k, nsav, mm) * nact(k, imode) +
                              raercol_cw(k, nsav, mm) * nact(k, imode);
            srcn += mam4::max(zero, tmpa);
          }
        }
        qcld(k) =
            explmix(qncld(km1), qncld(k), qncld(kp1), srcn, eddy_diff_kp(k),
                    eddy_diff_km(k), overlapp(k), overlapm(k), dtmix);
        for (int imode = 0; imode < nmodes; imode++) {
          for (int lspec = 0; lspec < AeroConfig::nspec_amode(imode) + 1;
               lspec++) {
            const int mm = AeroConfig::mam_idx(imode, lspec);
            Real source = 0;
            if (k < pver - 1) {
              const Real act = lspec ? mact(k, imode) : nact(k, imode);
              source = act * raercol(kp1, nsav, mm);
            } else {
              const Real tmpa = raercol(k, nsav, mm) * nact(k, imode) +
                                raercol_cw(k, nsav, mm) * nact(k, imode);
              source = mam4::max(zero, tmpa);
            }
            raercol_cw(k, nnew, mm) =
                explmix(raercol_cw(km1, nsav, mm), raercol_cw(k, nsav, mm),
                        raercol_cw(kp1, nsav, mm), source, eddy_diff_kp(k),
                        eddy_diff_km(k), overlapp(k), overlapm(k), dtmix);
            raercol(k, nnew, mm) =
                explmix(raercol(km1, nsav, mm), raercol(k, nsav, mm),
                        raercol(kp1, nsav, mm), source, eddy_diff_kp(k),
                        eddy_diff_km(k), overlapp(k), overlapm(k), dtmix,
                        raercol_cw(km1, nsav, mm), raercol_cw(kp1, nsav, mm));
          }
        }
      });
  team.team_barrier();
}

// ---------------------------------------------------------------------------
// Scalar tridiagonal families and solver dispatch
// ---------------------------------------------------------------------------

// Bands of A_C (cloud_family = true) or A_I (false) over the full column.
// Rows above top_lev are identity rows; their coupling to the active range is
// exactly zero because rate_up(top_lev) = 0. EKAT convention: dl(k) multiplies
// x(k-1) in row k, du(k) multiplies x(k+1) in row k.
KOKKOS_INLINE_FUNCTION
void assemble_bands(const ThreadTeam &team, const int top_lev, const Real h,
                    const bool cloud_family, const ColumnView &eddy_diff_kp,
                    const ColumnView &eddy_diff_km, const ColumnView &overlapp,
                    const ColumnView &overlapm, const View1D &dl,
                    const View1D &d, const View1D &du) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, pver), [&](int k) {
    if (k < top_lev) {
      dl(k) = Real(0);
      d(k) = Real(1);
      du(k) = Real(0);
      return;
    }
    const Real a = rate_up(k, top_lev, eddy_diff_km);
    const Real b = rate_down(k, eddy_diff_kp);
    d(k) = Real(1) + h * (a + b);
    dl(k) = -h * a * (cloud_family ? overlapm(k) : Real(1));
    du(k) = -h * b * (cloud_family ? overlapp(k) : Real(1));
  });
  team.team_barrier();
}

// Solve A X = B in place with the backend EKAT provides for this execution
// space (SHOC pattern): bit-for-bit Thomas when EKAT_DEFAULT_BFB is set,
// cyclic reduction on GPU builds, serial Thomas under Kokkos::single on CPU.
template <typename RhsView>
KOKKOS_INLINE_FUNCTION void
solve_tridiagonal(const ThreadTeam &team, const View1D &dl, const View1D &d,
                  const View1D &du, const RhsView &X) {
#ifdef EKAT_DEFAULT_BFB
  ekat::tridiag::bfb(team, dl, d, du, X);
#elif defined(MAM4XX_ENABLE_GPU)
  ekat::tridiag::cr(team, dl, d, du, X);
#else
  Kokkos::single(Kokkos::PerTeam(team),
                 [&]() { ekat::tridiag::thomas(dl, d, du, X); });
#endif
  team.team_barrier();
}

// Inspect a solved RHS before any dependent stage reads it. A negative is
// rejected even at roundoff scale; no clipping or ledger repair is performed.
KOKKOS_INLINE_FUNCTION
bool check_candidate_rhs(const ThreadTeam &team, const int top_lev,
                         const View2D &rhs, const int first_rhs,
                         int &negative_count) {
  int nonfinite = 0, negative = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, top_lev, pver),
      [&](int k, int &acc) {
        for (int j = first_rhs; j < rhs.extent_int(1); ++j)
          if (!is_finite(rhs(k, j)))
            ++acc;
      },
      nonfinite);
  Kokkos::parallel_reduce(
      Kokkos::TeamVectorRange(team, top_lev, pver),
      [&](int k, int &acc) {
        for (int j = first_rhs; j < rhs.extent_int(1); ++j)
          if (rhs(k, j) < Real(0))
            ++acc;
      },
      negative);
  team.team_barrier();
  negative_count = negative;
  return nonfinite == 0 && negative == 0;
}

// ---------------------------------------------------------------------------
// IMEX-Euler substep: candidate state into workspace, state views untouched
// ---------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
bool imex_substep(const ThreadTeam &team, const Real h, const int top_lev,
                  const int nsav, const View2D &nact, const View2D &mact,
                  const ColumnView &qcld, const View3D &raercol,
                  const View3D &raercol_cw, const ColumnView &eddy_diff_kp,
                  const ColumnView &eddy_diff_km, const ColumnView &overlapp,
                  const ColumnView &overlapm, const ImplicitWorkspace &ws,
                  int *negative_count = nullptr) {
  int negative = 0;
  if (negative_count)
    *negative_count = 0;
  // cloud-borne and droplet right-hand sides: old state plus explicit source
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, pver), [&](int k) {
    if (k < top_lev) {
      for (int j = 0; j < ncloud_rhs; ++j)
        ws.rhs_cloud(k, j) = Real(0);
      return;
    }
    const int kp1 = mam4::min(k + 1, pver - 1);
    Real srcn = Real(0);
    for (int imode = 0; imode < nmodes; ++imode) {
      const int mm0 = AeroConfig::mam_idx(imode, 0);
      srcn += nact(k, imode) * raercol(kp1, nsav, mm0);
      for (int lspec = 0; lspec < AeroConfig::nspec_amode(imode) + 1; ++lspec) {
        const int mm = AeroConfig::mam_idx(imode, lspec);
        const Real act = lspec ? mact(k, imode) : nact(k, imode);
        const Real source = act * raercol(kp1, nsav, mm);
        ws.rhs_cloud(k, 1 + mm) = raercol_cw(k, nsav, mm) + h * source;
      }
    }
    ws.rhs_cloud(k, 0) = qcld(k) + h * srcn;
  });
  team.team_barrier();
  assemble_bands(team, top_lev, h, true, eddy_diff_kp, eddy_diff_km, overlapp,
                 overlapm, ws.dl, ws.d, ws.du);
  solve_tridiagonal(team, ws.dl, ws.d, ws.du, ws.rhs_cloud);
  if (!check_candidate_rhs(team, top_lev, ws.rhs_cloud, 0, negative)) {
    if (negative_count)
      *negative_count = negative;
    return false;
  }

  // interstitial right-hand sides: old state, explicit sink, return at C^{n+1}
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, pver), [&](int k) {
    if (k < top_lev) {
      for (int mm = 0; mm < ncnst_tot; ++mm)
        ws.rhs_inter(k, mm) = Real(0);
      return;
    }
    const int kp1 = mam4::min(k + 1, pver - 1);
    const int km1 = mam4::max(k - 1, top_lev);
    const Real a = rate_up(k, top_lev, eddy_diff_km);
    const Real b = rate_down(k, eddy_diff_kp);
    const Real ret_up = h * a * (Real(1) - overlapm(k));
    const Real ret_down = h * b * (Real(1) - overlapp(k));
    for (int imode = 0; imode < nmodes; ++imode) {
      for (int lspec = 0; lspec < AeroConfig::nspec_amode(imode) + 1; ++lspec) {
        const int mm = AeroConfig::mam_idx(imode, lspec);
        const Real act = lspec ? mact(k, imode) : nact(k, imode);
        const Real source = act * raercol(kp1, nsav, mm);
        ws.rhs_inter(k, mm) = raercol(k, nsav, mm) - h * source +
                              ret_up * ws.rhs_cloud(km1, 1 + mm) +
                              ret_down * ws.rhs_cloud(kp1, 1 + mm);
      }
    }
  });
  team.team_barrier();
  assemble_bands(team, top_lev, h, false, eddy_diff_kp, eddy_diff_km, overlapp,
                 overlapm, ws.dl, ws.d, ws.du);
  solve_tridiagonal(team, ws.dl, ws.d, ws.du, ws.rhs_inter);
  const bool ok = check_candidate_rhs(team, top_lev, ws.rhs_inter, 0, negative);
  if (negative_count)
    *negative_count = negative;
  return ok;
}

// ---------------------------------------------------------------------------
// Backward Euler: guarded 2-by-2 block-Thomas per family, then droplets
// ---------------------------------------------------------------------------

// 2-by-2 helpers on (a00, a01, a10, a11).
struct Mat2 {
  Real a00, a01, a10, a11;
};
KOKKOS_INLINE_FUNCTION Mat2 mat2_mul(const Mat2 &x, const Mat2 &y) {
  return {x.a00 * y.a00 + x.a01 * y.a10, x.a00 * y.a01 + x.a01 * y.a11,
          x.a10 * y.a00 + x.a11 * y.a10, x.a10 * y.a01 + x.a11 * y.a11};
}
KOKKOS_INLINE_FUNCTION void mat2_apply(const Mat2 &m, const Real x0,
                                       const Real x1, Real &y0, Real &y1) {
  y0 = m.a00 * x0 + m.a01 * x1;
  y1 = m.a10 * x0 + m.a11 * x1;
}

// Blocks of the BE family at level k for activation rate act_k, in the
// unknown ordering x_k = (C_k, I_k). Backward Euler on the production
// stencil gives L_k = -h a_k [[oA,0],[1-oA,1]], B_k = lambda I,
// U_k = [[-h b oB, -h act],[-h b (1-oB), -h b + h act]].
// With act <= b (the activation cap) every off-diagonal entry is <= 0, and
// with weights w_k = 1/zn(k) the matrix is weighted column diagonally
// dominant (w_k b_k = w_{k+1} a_{k+1}), so it is a nonsingular M-matrix.
KOKKOS_INLINE_FUNCTION
void be_blocks(const int k, const int top_lev, const Real h, const Real act,
               const ColumnView &eddy_diff_kp, const ColumnView &eddy_diff_km,
               const ColumnView &overlapp, const ColumnView &overlapm, Mat2 &L,
               Real &lambda, Mat2 &U) {
  const Real a = rate_up(k, top_lev, eddy_diff_km);
  const Real b = rate_down(k, eddy_diff_kp);
  const Real oA = overlapm(k);
  const Real oB = overlapp(k);
  lambda = Real(1) + h * (a + b);
  L = {-h * a * oA, Real(0), -h * a * (Real(1) - oA), -h * a};
  U = {-h * b * oB, -h * act, -h * b * (Real(1) - oB), -h * b + h * act};
}

// Guarded inverse of a 2-by-2 pivot. In exact arithmetic the M-matrix
// structure gives det >= 1; the guard rejects nonfinite or degenerate pivots at
// |det| <= 100 eps ||D||_inf^2.
KOKKOS_INLINE_FUNCTION
bool invert_pivot(const Mat2 &D, Mat2 &Dinv) {
  const Real det = D.a00 * D.a11 - D.a01 * D.a10;
  const Real row0 = Kokkos::fabs(D.a00) + Kokkos::fabs(D.a01);
  const Real row1 = Kokkos::fabs(D.a10) + Kokkos::fabs(D.a11);
  const Real nrm = mam4::max(row0, row1);
  const Real threshold = Real(100) * machine_epsilon() * nrm * nrm;
  if (!is_finite(det) || !is_finite(nrm) || !(Kokkos::fabs(det) > threshold))
    return false;
  Dinv = {D.a11 / det, -D.a01 / det, -D.a10 / det, D.a00 / det};
  return is_finite(Dinv.a00) && is_finite(Dinv.a01) && is_finite(Dinv.a10) &&
         is_finite(Dinv.a11);
}

// One family: f < nmodes is the number constituent of mode f; f >= nmodes is
// the mass species of mode f - nmodes. Serial over levels within one thread.
// Reads the old state from the nsav buffer and writes the candidate C into
// ws.rhs_cloud(k, 1+mm) and I into ws.rhs_inter(k, mm). Returns false on a
// guarded pivot failure.
KOKKOS_INLINE_FUNCTION
bool be_family_solve(const int f, const Real h, const int top_lev,
                     const int nsav, const View2D &nact, const View2D &mact,
                     const View3D &raercol, const View3D &raercol_cw,
                     const ColumnView &eddy_diff_kp,
                     const ColumnView &eddy_diff_km, const ColumnView &overlapp,
                     const ColumnView &overlapm, const ImplicitWorkspace &ws) {
  const bool is_mass = f >= nmodes;
  const int imode = is_mass ? f - nmodes : f;
  const int lspec_begin = is_mass ? 1 : 0;
  const int lspec_end = is_mass ? AeroConfig::nspec_amode(imode) + 1 : 1;

  // load right-hand sides (old state)
  for (int k = top_lev; k < pver; ++k) {
    for (int lspec = lspec_begin; lspec < lspec_end; ++lspec) {
      const int mm = AeroConfig::mam_idx(imode, lspec);
      ws.rhs_cloud(k, 1 + mm) = raercol_cw(k, nsav, mm);
      ws.rhs_inter(k, mm) = raercol(k, nsav, mm);
    }
  }

  // forward elimination fused with the right-hand-side sweep
  Mat2 Dinv_prev{Real(0), Real(0), Real(0), Real(0)};
  Mat2 U_prev{Real(0), Real(0), Real(0), Real(0)};
  for (int k = top_lev; k < pver; ++k) {
    const Real act = is_mass ? mact(k, imode) : nact(k, imode);
    Mat2 L, U;
    Real lambda;
    be_blocks(k, top_lev, h, act, eddy_diff_kp, eddy_diff_km, overlapp,
              overlapm, L, lambda, U);
    Mat2 D{lambda, Real(0), Real(0), lambda};
    if (k > top_lev) {
      const Mat2 E = mat2_mul(L, Dinv_prev); // L_k D_{k-1}^{-1}
      const Mat2 EU = mat2_mul(E, U_prev);
      D = {lambda - EU.a00, -EU.a01, -EU.a10, lambda - EU.a11};
      for (int lspec = lspec_begin; lspec < lspec_end; ++lspec) {
        const int mm = AeroConfig::mam_idx(imode, lspec);
        Real e0, e1;
        mat2_apply(E, ws.rhs_cloud(k - 1, 1 + mm), ws.rhs_inter(k - 1, mm), e0,
                   e1);
        ws.rhs_cloud(k, 1 + mm) -= e0;
        ws.rhs_inter(k, mm) -= e1;
      }
    }
    Mat2 Dinv;
    if (!invert_pivot(D, Dinv))
      return false;
    ws.be_factor(k, f, 0) = Dinv.a00;
    ws.be_factor(k, f, 1) = Dinv.a01;
    ws.be_factor(k, f, 2) = Dinv.a10;
    ws.be_factor(k, f, 3) = Dinv.a11;
    Dinv_prev = Dinv;
    U_prev = U;
  }

  // back substitution
  for (int k = pver - 1; k >= top_lev; --k) {
    const Mat2 Dinv{ws.be_factor(k, f, 0), ws.be_factor(k, f, 1),
                    ws.be_factor(k, f, 2), ws.be_factor(k, f, 3)};
    Mat2 U{Real(0), Real(0), Real(0), Real(0)};
    if (k < pver - 1) {
      Mat2 L;
      Real lambda;
      const Real act = is_mass ? mact(k, imode) : nact(k, imode);
      be_blocks(k, top_lev, h, act, eddy_diff_kp, eddy_diff_km, overlapp,
                overlapm, L, lambda, U);
    }
    for (int lspec = lspec_begin; lspec < lspec_end; ++lspec) {
      const int mm = AeroConfig::mam_idx(imode, lspec);
      Real y0 = ws.rhs_cloud(k, 1 + mm);
      Real y1 = ws.rhs_inter(k, mm);
      if (k < pver - 1) {
        Real u0, u1;
        mat2_apply(U, ws.rhs_cloud(k + 1, 1 + mm), ws.rhs_inter(k + 1, mm), u0,
                   u1);
        y0 -= u0;
        y1 -= u1;
      }
      Real x0, x1;
      mat2_apply(Dinv, y0, y1, x0, x1);
      ws.rhs_cloud(k, 1 + mm) = x0;
      ws.rhs_inter(k, mm) = x1;
    }
  }
  return true;
}

// Backward Euler substep. Stage 1: the eight aerosol block families in
// parallel over team threads. Stage 2 (after a barrier, because its source
// needs the solved modal-number state): the droplet scalar solve with A_C.
// Returns false if any family hit a guarded pivot failure.
KOKKOS_INLINE_FUNCTION
bool be_substep(const ThreadTeam &team, const Real h, const int top_lev,
                const int nsav, const View2D &nact, const View2D &mact,
                const ColumnView &qcld, const View3D &raercol,
                const View3D &raercol_cw, const ColumnView &eddy_diff_kp,
                const ColumnView &eddy_diff_km, const ColumnView &overlapp,
                const ColumnView &overlapm, const ImplicitWorkspace &ws,
                int *negative_count = nullptr) {
  if (negative_count)
    *negative_count = 0;
  int failures = 0;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team, nfamilies),
      [&](const int f, int &acc) {
        if (!be_family_solve(f, h, top_lev, nsav, nact, mact, raercol,
                             raercol_cw, eddy_diff_kp, eddy_diff_km, overlapp,
                             overlapm, ws))
          ++acc;
      },
      failures);
  team.team_barrier();
  if (failures != 0)
    return false;
  int negative_cloud = 0, negative_inter = 0;
  const bool cloud_ok =
      check_candidate_rhs(team, top_lev, ws.rhs_cloud, 1, negative_cloud);
  const bool inter_ok =
      check_candidate_rhs(team, top_lev, ws.rhs_inter, 0, negative_inter);
  if (negative_count)
    *negative_count = negative_cloud + negative_inter;
  if (!cloud_ok || !inter_ok)
    return false;

  // droplets: source from the solved new-level interstitial modal numbers
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, pver), [&](int k) {
    if (k < top_lev) {
      ws.rhs_drop(k, 0) = Real(0);
      return;
    }
    const int kp1 = mam4::min(k + 1, pver - 1);
    Real srcn = Real(0);
    for (int imode = 0; imode < nmodes; ++imode) {
      const int mm0 = AeroConfig::mam_idx(imode, 0);
      srcn += nact(k, imode) * ws.rhs_inter(kp1, mm0);
    }
    ws.rhs_drop(k, 0) = qcld(k) + h * srcn;
  });
  team.team_barrier();
  assemble_bands(team, top_lev, h, true, eddy_diff_kp, eddy_diff_km, overlapp,
                 overlapm, ws.dl, ws.d, ws.du);
  solve_tridiagonal(team, ws.dl, ws.d, ws.du, ws.rhs_drop);
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 0, pver),
                       [&](int k) { ws.rhs_cloud(k, 0) = ws.rhs_drop(k, 0); });
  team.team_barrier();
  int negative = 0;
  const bool ok = check_candidate_rhs(team, top_lev, ws.rhs_drop, 0, negative);
  if (negative_count)
    *negative_count = negative;
  return ok;
}

// ---------------------------------------------------------------------------
// Acceptance checks and commit
// ---------------------------------------------------------------------------

// Copy the accepted raw candidate without clipping or repair.
KOKKOS_INLINE_FUNCTION
void commit_candidate(const ThreadTeam &team, const int top_lev, const int nnew,
                      const ColumnView &qcld, const View3D &raercol,
                      const View3D &raercol_cw, const ImplicitWorkspace &ws) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, top_lev, pver),
                       [&](int k) {
                         qcld(k) = ws.rhs_cloud(k, 0);
                         for (int mm = 0; mm < ncnst_tot; ++mm) {
                           raercol_cw(k, nnew, mm) = ws.rhs_cloud(k, 1 + mm);
                           raercol(k, nnew, mm) = ws.rhs_inter(k, mm);
                         }
                       });
  team.team_barrier();
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

// Advance the column over dtmicro with the selected implicit integrator and
// substep policy, falling back to Forward Euler on rejection. On entry the
// controller has produced dtmix and nsubmix; nsav/nnew are the FE buffer
// indices and are updated exactly as the FE loop would. update_from_explmix
// has already checked the input, geometry, and workspace before the
// controller arithmetic. A valid operator outside the qualified coefficient
// range takes the full FE path; an invalid activation operator or a nonfinite
// FE result is an unrecoverable error.
KOKKOS_INLINE_FUNCTION
void advance(const ThreadTeam &team, const Real dtmicro, const Real dtmix,
             const int nsubmix, const int top_lev, const ColumnView &zn,
             const View2D &nact, const View2D &mact, const ColumnView &qcld,
             const ColumnView &qncld, const View3D &raercol,
             const View3D &raercol_cw, int &nsav, int &nnew,
             const ColumnView &eddy_diff_kp, const ColumnView &eddy_diff_km,
             const ColumnView &overlapp, const ColumnView &overlapm,
             const MixingOptions &options, const ImplicitWorkspace &ws,
             MixingStatus *status) {
  MixingStatus st;

  st.input_negative_count =
      count_negative_input(team, top_lev, nsav, qcld, raercol, raercol_cw);

  const bool single = options.policy == ImplicitSubstepPolicy::SingleStep;
  const int n_impl = single ? 1 : nsubmix;
  const Real h = single ? dtmicro : dtmix;

  EKAT_KERNEL_REQUIRE_MSG(ws.valid(),
                          "ndrop implicit mixing: workspace not allocated");
  EKAT_KERNEL_REQUIRE_MSG(
      activation_preconditions_hold(team, top_lev, nact, mact, eddy_diff_kp),
      "ndrop implicit mixing: invalid activation operator");
  bool run_full_fe =
      !within_implicit_range(team, top_lev, h, eddy_diff_kp, eddy_diff_km);
  if (run_full_fe)
    st.fallback_count = 1;

  if (!run_full_fe) {
    for (int isub = 0; isub < n_impl; ++isub) {
      if (isub > 0) {
        const int ntemp = nsav;
        nsav = nnew;
        nnew = ntemp;
      }
      bool solver_ok;
      int negative = 0;
      if (options.integrator == MixingIntegrator::IMEX) {
        solver_ok = imex_substep(
            team, h, top_lev, nsav, nact, mact, qcld, raercol, raercol_cw,
            eddy_diff_kp, eddy_diff_km, overlapp, overlapm, ws, &negative);
      } else {
        solver_ok = be_substep(team, h, top_lev, nsav, nact, mact, qcld,
                               raercol, raercol_cw, eddy_diff_kp, eddy_diff_km,
                               overlapp, overlapm, ws, &negative);
      }
      const bool accept = solver_ok;
      if (accept) {
        commit_candidate(team, top_lev, nnew, qcld, raercol, raercol_cw, ws);
        ++st.nsubmix_executed;
      } else {
        ++st.fallback_count;
        if (negative != 0)
          ++st.material_negative_count;
        if (single) {
          // discard the implicit step; nsav/nnew are still the entry values
          run_full_fe = true;
          break;
        }
        // ControllerCount: redo this substep with FE at the same h, which is
        // FE-stable by construction. qcld is still the old state.
        forward_euler_substep(team, dtmix, top_lev, nsav, nnew, nact, mact,
                              qcld, qncld, raercol, raercol_cw, eddy_diff_kp,
                              eddy_diff_km, overlapp, overlapm);
        EKAT_KERNEL_REQUIRE_MSG(
            count_nonfinite_input(team, top_lev, nnew, qcld, raercol,
                                  raercol_cw) == 0,
            "ndrop implicit mixing: nonfinite Forward Euler redo result");
        ++st.nsubmix_executed;
      }
      team.team_barrier();
    }
  }

  if (run_full_fe) {
    // full Forward Euler controller loop from the intact nsav state
    for (int isub = 0; isub < nsubmix; ++isub) {
      if (isub > 0) {
        const int ntemp = nsav;
        nsav = nnew;
        nnew = ntemp;
      }
      forward_euler_substep(team, dtmix, top_lev, nsav, nnew, nact, mact, qcld,
                            qncld, raercol, raercol_cw, eddy_diff_kp,
                            eddy_diff_km, overlapp, overlapm);
      // Stop before a later redo step can consume a nonfinite result.
      EKAT_KERNEL_REQUIRE_MSG(
          count_nonfinite_input(team, top_lev, nnew, qcld, raercol,
                                raercol_cw) == 0,
          "ndrop implicit mixing: nonfinite Forward Euler result");
    }
    st.nsubmix_executed = nsubmix;
  }

  if (status != nullptr) {
    // every team thread holds identical values; the write is benign
    *status = st;
  }
}

} // namespace implicit_mixing
} // namespace ndrop
} // namespace mam4

#endif
