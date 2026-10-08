// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <catch2/catch.hpp>
#include <ekat_fpe.hpp>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

using mam4::Real;
using namespace mam4::gas_chemistry;

struct SolverCase {
  std::array<Real, gas_pcnst> state = {};
  std::array<Real, rxntot> rates = {};
  std::array<Real, gas_pcnst> het = {};
  std::array<Real, extcnt> forcing = {};
  Real dt = 1;
};

struct SolverOutput {
  std::array<Real, gas_pcnst> state = {};
  std::array<Real, clscnt4> production = {};
  std::array<Real, clscnt4> loss = {};
  ImpSolResult result;
};

// These fixtures deliberately overflow so that we can test the solver's
// nonfinite-result status. Restore the test session's overflow trap afterward.
struct ScopedOverflowTrapDisable {
  const bool was_enabled = (ekat::get_enabled_fpes() & FE_OVERFLOW) != 0;

  ScopedOverflowTrapDisable() {
    if (was_enabled) {
      ekat::disable_fpes(FE_OVERFLOW);
    }
  }
  ~ScopedOverflowTrapDisable() {
    if (was_enabled) {
      ekat::enable_fpes(FE_OVERFLOW);
    }
  }
};

SolverOutput run_on_device(const SolverCase &input) {
  using RealView = mam4::DeviceType::view_1d<Real>;
  using ResultView = mam4::DeviceType::view_1d<ImpSolResult>;

  RealView state("gas_state", gas_pcnst);
  RealView rates("gas_rates", rxntot);
  RealView het("gas_het", gas_pcnst);
  RealView forcing("gas_forcing", extcnt);
  RealView production("gas_production", clscnt4);
  RealView loss("gas_loss", clscnt4);
  ResultView result("gas_result", 1);

  auto h_state = Kokkos::create_mirror_view(state);
  auto h_rates = Kokkos::create_mirror_view(rates);
  auto h_het = Kokkos::create_mirror_view(het);
  auto h_forcing = Kokkos::create_mirror_view(forcing);
  for (int j = 0; j < gas_pcnst; ++j) {
    h_state(j) = input.state[j];
    h_het(j) = input.het[j];
  }
  for (int i = 0; i < rxntot; ++i) {
    h_rates(i) = input.rates[i];
  }
  for (int i = 0; i < extcnt; ++i) {
    h_forcing(i) = input.forcing[i];
  }
  Kokkos::deep_copy(state, h_state);
  Kokkos::deep_copy(rates, h_rates);
  Kokkos::deep_copy(het, h_het);
  Kokkos::deep_copy(forcing, h_forcing);
  Kokkos::deep_copy(production, Real(-1));
  Kokkos::deep_copy(loss, Real(-1));

  const Real dt = input.dt;
  Kokkos::parallel_for(
      "direct_gas_chemistry_test", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        auto local_state = state;
        ImpSolResult local_result;
        imp_sol(local_state, rates.data(), het.data(), forcing.data(), dt,
                production.data(), loss.data(), local_result);
        result(0) = local_result;
      });
  Kokkos::fence();

  Kokkos::deep_copy(h_state, state);
  auto h_production = Kokkos::create_mirror_view(production);
  auto h_loss = Kokkos::create_mirror_view(loss);
  auto h_result = Kokkos::create_mirror_view(result);
  Kokkos::deep_copy(h_production, production);
  Kokkos::deep_copy(h_loss, loss);
  Kokkos::deep_copy(h_result, result);

  SolverOutput output;
  for (int j = 0; j < gas_pcnst; ++j) {
    output.state[j] = h_state(j);
  }
  for (int k = 0; k < clscnt4; ++k) {
    output.production[k] = h_production(k);
    output.loss[k] = h_loss(k);
  }
  output.result = h_result(0);
  return output;
}

// Test-only reference assembled from the documented chemistry equations,
// without using production linmat, indprd, or the ordered-solve helper.
std::array<Real, clscnt4> dense_backward_euler_reference(
    const SolverCase &input) {
  using Row = std::array<long double, clscnt4>;
  std::array<Row, clscnt4> matrix = {};
  std::array<long double, clscnt4> source = {};
  std::array<long double, clscnt4> rhs = {};
  const auto &r = input.rates;
  const auto &h = input.het;
  const auto &f = input.forcing;

  matrix[0][0] = -(static_cast<long double>(h[1]) + r[0] + r[2]);
  matrix[1][1] = -static_cast<long double>(h[2]);
  matrix[1][2] = r[3];
  matrix[2][2] = -(static_cast<long double>(h[3]) + r[3]);
  matrix[2][3] = static_cast<long double>(r[4]) + 0.5L * r[5] + r[6];
  matrix[3][3] = -(static_cast<long double>(h[4]) + r[4] + r[5] + r[6]);
  for (int k = 4; k < clscnt4; ++k) {
    matrix[k][k] = -static_cast<long double>(h[k + 1]);
  }
  source[0] = r[1];
  source[2] = f[0];
  source[4] = f[8];
  source[5] = f[1];
  source[12] = f[5];
  source[13] = f[2];
  source[17] = f[6];
  source[26] = f[3];
  source[27] = f[4];
  source[29] = f[7];

  const long double dt = input.dt;
  for (int row = 0; row < clscnt4; ++row) {
    rhs[row] = static_cast<long double>(input.state[row + 1]) + dt * source[row];
    for (int col = 0; col < clscnt4; ++col) {
      matrix[row][col] = (row == col ? 1.0L : 0.0L) - dt * matrix[row][col];
    }
  }
  for (int pivot = 0; pivot < clscnt4; ++pivot) {
    int best = pivot;
    for (int row = pivot + 1; row < clscnt4; ++row) {
      if (std::fabs(matrix[row][pivot]) > std::fabs(matrix[best][pivot])) {
        best = row;
      }
    }
    REQUIRE(matrix[best][pivot] != 0);
    std::swap(matrix[pivot], matrix[best]);
    std::swap(rhs[pivot], rhs[best]);
    for (int row = pivot + 1; row < clscnt4; ++row) {
      const long double factor = matrix[row][pivot] / matrix[pivot][pivot];
      for (int col = pivot; col < clscnt4; ++col) {
        matrix[row][col] -= factor * matrix[pivot][col];
      }
      rhs[row] -= factor * rhs[pivot];
    }
  }
  std::array<Real, clscnt4> answer = {};
  for (int row = clscnt4 - 1; row >= 0; --row) {
    long double value = rhs[row];
    for (int col = row + 1; col < clscnt4; ++col) {
      value -= matrix[row][col] * answer[col];
    }
    answer[row] = static_cast<Real>(value / matrix[row][row]);
  }
  return answer;
}

Real source_for_entry(const SolverCase &input, const int k) {
  const auto &f = input.forcing;
  switch (k) {
  case 0: return input.rates[1];
  case 2: return f[0];
  case 4: return f[8];
  case 5: return f[1];
  case 12: return f[5];
  case 13: return f[2];
  case 17: return f[6];
  case 26: return f[3];
  case 27: return f[4];
  case 29: return f[7];
  default: return 0;
  }
}

Real entry_scale(const Real a, const Real b, const Real c = 0) {
  return std::max({std::abs(a), std::abs(b), std::abs(c),
                   std::numeric_limits<Real>::min()});
}

SolverCase nominal_case() {
  SolverCase input;
  input.dt = 100;
  input.state[0] = -1e-6; // O3 is finite, signed, and outside this operator.
  for (int j = 1; j < gas_pcnst; ++j) {
    input.state[j] = (j == 13 || j == 18 || j == 26 || j == 30)
                         ? Real(1e8 + j * 1e6)
                         : Real(j * 1e-9);
    input.het[j] = Real((j % 4 + 1) * 1e-5);
  }
  input.rates = {Real(1e-4), Real(1e-10), Real(3e-4), Real(2e-4),
                 Real(1e-4), Real(2e-4), Real(3e-4)};
  input.forcing = {Real(1e-9), Real(1e-11), Real(2e-11),
                   Real(1e-11), Real(1e-11), Real(1e4),
                   Real(2e4), Real(3e4), Real(5e-11)};
  return input;
}

void require_atomic_failure(const SolverCase &input,
                            const ImpSolOutcome expected) {
  const auto output = run_on_device(input);
  REQUIRE(output.result.outcome == expected);
  REQUIRE_FALSE(output.result.success());
  for (int j = 0; j < gas_pcnst; ++j) {
    if (std::isnan(input.state[j])) {
      REQUIRE(std::isnan(output.state[j]));
    } else {
      REQUIRE(std::memcmp(&output.state[j], &input.state[j], sizeof(Real)) == 0);
    }
  }
  for (int k = 0; k < clscnt4; ++k) {
    REQUIRE(output.production[k] == 0);
    REQUIRE(output.loss[k] == 0);
  }
}

} // namespace

TEST_CASE("direct gas chemistry matches independent backward Euler",
          "[gas_chem]") {
  const auto input = nominal_case();
  const auto output = run_on_device(input);
  const auto reference = dense_backward_euler_reference(input);
  REQUIRE(output.result.success());
  REQUIRE(std::memcmp(&output.state[0], &input.state[0], sizeof(Real)) == 0);

  const Real tolerance = 128 * std::numeric_limits<Real>::epsilon();
  for (int k = 0; k < clscnt4; ++k) {
    const int j = k + 1;
    const Real scale = entry_scale(reference[k], input.state[j],
                                   input.dt * source_for_entry(input, k));
    REQUIRE(std::abs(output.state[j] - reference[k]) <= tolerance * scale);

    Real expected_production = source_for_entry(input, k);
    Real expected_loss = input.het[j] * output.state[j];
    if (k == 0) {
      expected_loss += (input.rates[0] + input.rates[2]) * output.state[j];
    } else if (k == 1) {
      expected_production += input.rates[3] * output.state[3];
    } else if (k == 2) {
      expected_production +=
          (input.rates[4] + Real(0.5) * input.rates[5] + input.rates[6]) *
          output.state[4];
      expected_loss += input.rates[3] * output.state[j];
    } else if (k == 3) {
      expected_loss += (input.rates[4] + input.rates[5] + input.rates[6]) *
                       output.state[j];
    }
    REQUIRE(std::abs(output.production[k] - expected_production) <=
            tolerance * entry_scale(output.production[k], expected_production));
    REQUIRE(std::abs(output.loss[k] - expected_loss) <=
            tolerance * entry_scale(output.loss[k], expected_loss));
    const Real residual = output.state[j] - input.state[j] -
                          input.dt * (output.production[k] - output.loss[k]);
    const Real residual_scale =
        std::abs(output.state[j]) + std::abs(input.state[j]) +
        input.dt * (std::abs(output.production[k]) + std::abs(output.loss[k])) +
        std::numeric_limits<Real>::min();
    REQUIRE(std::abs(residual) <= tolerance * residual_scale);
  }
}

TEST_CASE("direct gas chemistry preserves zero rates and handles timestep scales",
          "[gas_chem]") {
  SolverCase zero;
  zero.state = nominal_case().state;
  const auto unchanged = run_on_device(zero);
  REQUIRE(unchanged.result.success());
  for (int j = 0; j < gas_pcnst; ++j) {
    REQUIRE(unchanged.state[j] == zero.state[j]);
  }

  for (const Real dt : {Real(1e-8), Real(1e6)}) {
    auto input = nominal_case();
    input.dt = dt;
    const auto output = run_on_device(input);
    const auto reference = dense_backward_euler_reference(input);
    REQUIRE(output.result.success());
    const Real tolerance = 128 * std::numeric_limits<Real>::epsilon();
    for (int k = 0; k < clscnt4; ++k) {
      const int j = k + 1;
      REQUIRE(std::abs(output.state[j] - reference[k]) <=
              tolerance * entry_scale(reference[k], input.state[j]));
    }
  }
}

TEST_CASE("direct gas chemistry accepts repaired signed inputs", "[gas_chem]") {
  SolverCase input;
  input.state[0] = -2;
  input.state[3] = -1e-6; // SO2
  input.state[5] = -1e-6; // SOAG
  input.forcing[0] = 2e-6;
  input.forcing[8] = 2e-6;
  const auto output = run_on_device(input);
  REQUIRE(output.result.success());
  REQUIRE(output.state[3] == Approx(1e-6));
  REQUIRE(output.state[5] == Approx(1e-6));
  REQUIRE(std::memcmp(&output.state[0], &input.state[0], sizeof(Real)) == 0);

  input = SolverCase{};
  input.state[5] = 3e-6; // SOAG
  input.forcing[8] = -1e-6;
  const auto signed_source_output = run_on_device(input);
  REQUIRE(signed_source_output.result.success());
  REQUIRE(signed_source_output.state[5] == Approx(Real(2e-6)));
  REQUIRE(signed_source_output.production[4] == Approx(Real(-1e-6)));
  REQUIRE(signed_source_output.loss[4] == 0);
  REQUIRE(signed_source_output.state[5] - input.state[5] ==
          Approx(input.dt * (signed_source_output.production[4] -
                             signed_source_output.loss[4])));
  for (int j = 0; j < gas_pcnst; ++j) {
    if (j != 5) {
      REQUIRE(signed_source_output.state[j] == input.state[j]);
    }
  }
}

TEST_CASE("direct gas chemistry fails atomically", "[gas_chem]") {
  SolverCase input;
  input.dt = 0;
  require_atomic_failure(input, ImpSolOutcome::InvalidInput);

  input = SolverCase{};
  input.state[0] = std::numeric_limits<Real>::quiet_NaN();
  require_atomic_failure(input, ImpSolOutcome::InvalidInput);

  input = SolverCase{};
  input.rates[3] = -1;
  require_atomic_failure(input, ImpSolOutcome::InvalidInput);

  input = SolverCase{};
  {
    ScopedOverflowTrapDisable overflow_trap;
    input.dt = 2;
    input.het[5] = std::numeric_limits<Real>::max();
    require_atomic_failure(input, ImpSolOutcome::NonfiniteResult);
  }

  input = SolverCase{};
  input.state[5] = 1e-6;
  input.forcing[8] = -2e-6;
  require_atomic_failure(input, ImpSolOutcome::NegativeResult);

  input = SolverCase{};
  input.state[30] = 1;
  input.forcing[7] = -2;
  require_atomic_failure(input, ImpSolOutcome::NegativeResult);

  input = SolverCase{};
  {
    ScopedOverflowTrapDisable overflow_trap;
    input.dt = 2;
    input.forcing[8] = std::numeric_limits<Real>::max();
    require_atomic_failure(input, ImpSolOutcome::NonfiniteResult);
  }

  input = SolverCase{};
  {
    ScopedOverflowTrapDisable overflow_trap;
    input.dt = std::numeric_limits<Real>::min();
    input.state[1] = 10;
    input.het[1] = std::numeric_limits<Real>::max() / 2;
    require_atomic_failure(input, ImpSolOutcome::NonfiniteResult);
  }
}

TEST_CASE("direct backward Euler row rejects an unsafe denominator",
          "[gas_chem]") {
  Real updated = -1;
  const auto outcome =
      solve_backward_euler_row(Real(1), Real(0), Real(2), Real(1), updated);
  REQUIRE(outcome == ImpSolOutcome::UnsafeDenominator);
  REQUIRE(updated == Real(-1));
}
