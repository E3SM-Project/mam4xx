// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <mam4_test_config.hpp>

#include <array>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

// The expected-abort test uses POSIX fork() and waitpid() to isolate the
// solver abort and let the parent verify how the child terminated. These
// functions are not part of the C++ standard; no build-system availability
// check is implemented because they are assumed available on the systems where
// mam4xx tests typically run.
#include <sys/wait.h>
#include <unistd.h>

namespace {

using mam4::Real;
using namespace mam4::gas_chemistry;

struct ChemistryOutput {
  std::array<Real, gas_pcnst> state;
  ImpSolResult result;
};

template <typename VectorType>
KOKKOS_INLINE_FUNCTION ImpSolResult call_wrapper(VectorType &state,
                                                 const Real dt,
                                                 const Real soag_forcing) {
  Real photo_rates[mam4::mo_photo::phtcnt] = {};
  Real forcing[extcnt] = {};
  Real invariants[nfs] = {};
  Real heterogeneous_rates[gas_pcnst] = {};
  // Synthetic M=1 leaves forcing numerically unchanged by normalization;
  // zero oxidants and photolysis isolate the external SOAG source.
  invariants[indexm] = 1;
  forcing[8] = soag_forcing;
  return mam4::microphysics::gas_phase_chemistry(Real(300), dt, photo_rates,
                                                 forcing, invariants,
                                                 heterogeneous_rates, state);
}

[[maybe_unused]] ChemistryOutput
run_wrapper(const std::array<Real, gas_pcnst> &initial, const Real dt,
            const Real soag_forcing) {
  using StateView = mam4::DeviceType::view_1d<Real>;
  using ResultView = mam4::DeviceType::view_1d<ImpSolResult>;
  StateView state("wrapper_state", gas_pcnst);
  ResultView result("wrapper_result", 1);
  auto host_state = Kokkos::create_mirror_view(state);
  for (int j = 0; j < gas_pcnst; ++j) {
    host_state(j) = initial[j];
  }
  Kokkos::deep_copy(state, host_state);
  Kokkos::parallel_for(
      "wrapper_status", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
        auto local_state = state;
        result(0) = call_wrapper(local_state, dt, soag_forcing);
      });
  Kokkos::fence();
  const auto accepted =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);
  const auto status =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  ChemistryOutput output;
  output.result = status(0);
  for (int j = 0; j < gas_pcnst; ++j) {
    output.state[j] = accepted(j);
  }
  return output;
}

// Isolate Kokkos::abort in a child so Catch can continue running other tests.
// Call the wrapper on the host: GPU runtimes are not assumed safe to use
// after fork, so the child must not launch device kernels.
[[maybe_unused]] void
require_wrapper_abort(const std::array<Real, gas_pcnst> &initial, const Real dt,
                      const Real soag_forcing) {
  const pid_t child = fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    // Use SIGABRT's default action in the child so Catch's inherited SIGABRT
    // handler does not intercept the abort. The parent calls waitpid to check
    // that the child terminated because of SIGABRT.
    std::signal(SIGABRT, SIG_DFL);
    Real state[gas_pcnst];
    for (int j = 0; j < gas_pcnst; ++j) {
      state[j] = initial[j];
    }
    call_wrapper(state, dt, soag_forcing);
    // Skip inherited cleanup handlers and destructors in the forked child.
    std::_Exit(EXIT_SUCCESS);
  }
  int status = 0;
  REQUIRE(waitpid(child, &status, 0) == child);
  REQUIRE(WIFSIGNALED(status));
  REQUIRE(WTERMSIG(status) == SIGABRT);
}

} // namespace

TEST_CASE("gas chemistry preserves diagnostic outcome codes", "[gas_chem]") {
  using mam4::gas_chemistry::ImpSolOutcome;
  REQUIRE(static_cast<int>(ImpSolOutcome::Converged) == 0);
  REQUIRE(static_cast<int>(ImpSolOutcome::ConvergedAfterRetry) == 1);
  REQUIRE(static_cast<int>(ImpSolOutcome::InvalidInput) == 2);
  REQUIRE(static_cast<int>(ImpSolOutcome::NonfiniteIterate) == 3);
  REQUIRE(static_cast<int>(ImpSolOutcome::CutLimitExhausted) == 4);
  REQUIRE(static_cast<int>(ImpSolOutcome::MaximumStepsExhausted) == 5);
  REQUIRE(static_cast<int>(ImpSolOutcome::NonfiniteResult) == 6);
  REQUIRE(static_cast<int>(ImpSolOutcome::UnsafeDenominator) == 7);
  REQUIRE(static_cast<int>(ImpSolOutcome::NegativeResult) == 8);
}

TEST_CASE("configured gas chemistry returns failure accounting", "[gas_chem]") {
  using namespace mam4::gas_chemistry;
  using mam4::Real;
  std::array<Real, gas_pcnst> initial;
  for (int j = 0; j < gas_pcnst; ++j) {
    initial[j] = Real(j + 1) * Real(1e-9);
  }
  Real dt = 1;
  Real forcing = 0;
  SECTION("invalid interval") { dt = -1; }
  SECTION("negative endpoint") {
    initial[5] = Real(1e-6);
    forcing = Real(-2e-6);
  }

  // Direct Debug failures terminate through Kokkos::abort. The other
  // configurations return diagnostics that the assertions below inspect.
#if !defined(MAM4XX_GAS_CHEMISTRY_SOLVER_LEGACY) && !defined(NDEBUG)
  require_wrapper_abort(initial, dt, forcing);
#else
  const auto output = run_wrapper(initial, dt, forcing);
  const auto result = output.result;
  const bool direct_selected =
      std::string(MAM4_TEST_GAS_CHEMISTRY_SOLVER) == "direct";
  REQUIRE_FALSE(result.success());
  REQUIRE(result.requested_interval == dt);

  if (dt < 0) {
    REQUIRE(result.outcome == ImpSolOutcome::InvalidInput);
    REQUIRE(result.accepted_steps == 0);
    REQUIRE(result.accepted_interval == 0);
    REQUIRE(result.cut_count == 0);
    REQUIRE(result.failed_attempts == (direct_selected ? 1 : 0));
    for (int j = 0; j < gas_pcnst; ++j) {
      REQUIRE(std::memcmp(&output.state[j], &initial[j], sizeof(Real)) == 0);
    }
  } else if (direct_selected) {
    REQUIRE(result.outcome == ImpSolOutcome::NegativeResult);
    REQUIRE(result.failed_attempts == 1);
    REQUIRE(result.cut_count == 0);
    REQUIRE(result.accepted_steps == 0);
    REQUIRE(result.accepted_interval == 0);
    for (int j = 0; j < gas_pcnst; ++j) {
      REQUIRE(std::memcmp(&output.state[j], &initial[j], sizeof(Real)) == 0);
    }
  } else {
    // Accepted legacy substeps remain committed when a later retry fails.
    REQUIRE(result.outcome == ImpSolOutcome::CutLimitExhausted);
    REQUIRE(result.failed_attempts > 0);
    REQUIRE(result.cut_count == 5);
    REQUIRE(result.accepted_steps > 0);
    REQUIRE(result.accepted_interval > 0);
    REQUIRE(result.accepted_interval < dt);
    REQUIRE(output.state[5] >= 0);
    REQUIRE(output.state[5] < initial[5]);
    const Real expected = initial[5] + result.accepted_interval * forcing;
    const Real tolerance = Real(64) * std::numeric_limits<Real>::epsilon();
    REQUIRE(std::abs(output.state[5] - expected) <=
            tolerance * std::abs(initial[5]));
    REQUIRE(output.state[0] == initial[0]);
  }
#endif
}

TEST_CASE("configured gas chemistry solver runs three timesteps",
          "[gas_chem][smoke]") {
  using mam4::Real;
  using namespace mam4::gas_chemistry;

  // The expected option comes directly from the CMake cache, independently
  // of the production macros that select the wrapper's solver call.
#if defined(MAM4XX_GAS_CHEMISTRY_SOLVER_LEGACY)
  REQUIRE(std::string(MAM4_TEST_GAS_CHEMISTRY_SOLVER) == "legacy");
#else
  REQUIRE(std::string(MAM4_TEST_GAS_CHEMISTRY_SOLVER) == "direct");
#endif

  constexpr int steps = 3;
  using RealView = mam4::DeviceType::view_1d<Real>;
  RealView history("gas_chemistry_smoke_history", (steps + 1) * gas_pcnst);

  Kokkos::parallel_for(
      "gas_chemistry_smoke", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        Real state[gas_pcnst];
        Real het_rates[gas_pcnst] = {};
        for (int j = 0; j < gas_pcnst; ++j) {
          // Synthetic values in the existing chemistry work-entry basis.
          state[j] = Real(j + 1) * Real(1e-9);
          het_rates[j] = j == o3_idx ? Real(0) : Real(1e-5); // s^-1
          history(j) = state[j];
        }

        Real photo_rates[mam4::mo_photo::phtcnt] = {};
        photo_rates[0] = Real(1e-5); // H2O2 photolysis frequency [s^-1]
        Real forcing[extcnt] = {};
        forcing[0] = Real(1e6); // SO2 source [molecules cm^-3 s^-1]
        forcing[8] = Real(1e6); // SOAG source [molecules cm^-3 s^-1]

        // Prescribed M, N2, O2, H2O, OH, NO3, HO2, O3 [molecules cm^-3].
        Real invariants[nfs] = {Real(2.5e19), Real(1.95e19), Real(5.25e18),
                                Real(1e17),   Real(1e6),     Real(1e7),
                                Real(1e8),    Real(1e12)};

        for (int step = 1; step <= steps; ++step) {
          mam4::microphysics::gas_phase_chemistry(Real(300), Real(60),
                                                  photo_rates, forcing,
                                                  invariants, het_rates, state);
          for (int j = 0; j < gas_pcnst; ++j) {
            history(step * gas_pcnst + j) = state[j];
          }
        }
      });
  Kokkos::fence();

  auto host_history = Kokkos::create_mirror_view(history);
  Kokkos::deep_copy(host_history, history);
  for (int step = 1; step <= steps; ++step) {
    CAPTURE(step);
    for (int j = 0; j < gas_pcnst; ++j) {
      CAPTURE(j);
      REQUIRE(std::isfinite(host_history(step * gas_pcnst + j)));
      REQUIRE(host_history(step * gas_pcnst + j) >= 0);
    }
    REQUIRE(host_history(step * gas_pcnst + o3_idx) == host_history(o3_idx));
    // DMS has losses and no source, so every call must advance the state.
    REQUIRE(host_history(step * gas_pcnst + 4) <
            host_history((step - 1) * gas_pcnst + 4));
  }
}
