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
#include <string>
#include <vector>

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

enum class Solver { Direct, Legacy };

struct SolverOutput {
  std::array<Real, gas_pcnst> state = {};
  std::array<Real, clscnt4> production = {};
  std::array<Real, clscnt4> loss = {};
  ImpSolOutcome outcome = ImpSolOutcome::InvalidInput;
  ImpSolResult legacy_result;
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

SolverOutput run_on_device(const SolverCase &input,
                           const Solver solver = Solver::Direct,
                           const Real reference_tolerance = Real(1e-12)) {
  using RealView = mam4::DeviceType::view_1d<Real>;
  using OutcomeView = mam4::DeviceType::view_1d<ImpSolOutcome>;
  using ResultView = mam4::DeviceType::view_1d<ImpSolResult>;

  RealView state("gas_state", gas_pcnst);
  RealView rates("gas_rates", rxntot);
  RealView het("gas_het", gas_pcnst);
  RealView forcing("gas_forcing", extcnt);
  RealView production("gas_production", clscnt4);
  RealView loss("gas_loss", clscnt4);
  OutcomeView outcome("gas_outcome", 1);
  ResultView legacy_result("legacy_result", 1);

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
        // Both methods are compiled and receive identical input arrays.
        // The production selection macro is intentionally not used here.
        if (solver == Solver::Legacy) {
          Real epsilon[clscnt4];
          for (int k = 0; k < clscnt4; ++k) {
            epsilon[k] = reference_tolerance;
          }
          bool factor[legacy::itermax];
          for (int k = 0; k < legacy::itermax; ++k) {
            factor[k] = true;
          }
          legacy_result(0) = legacy::imp_sol(
              local_state, rates.data(), het.data(), forcing.data(), dt, factor,
              epsilon, production.data(), loss.data());
          outcome(0) = legacy_result(0).outcome;
        } else {
          outcome(0) =
              imp_sol(local_state, rates.data(), het.data(), forcing.data(), dt,
                      production.data(), loss.data());
        }
      });
  Kokkos::fence();

  Kokkos::deep_copy(h_state, state);
  auto h_production = Kokkos::create_mirror_view(production);
  auto h_loss = Kokkos::create_mirror_view(loss);
  auto h_outcome = Kokkos::create_mirror_view(outcome);
  Kokkos::deep_copy(h_production, production);
  Kokkos::deep_copy(h_loss, loss);
  Kokkos::deep_copy(h_outcome, outcome);

  SolverOutput output;
  for (int j = 0; j < gas_pcnst; ++j) {
    output.state[j] = h_state(j);
  }
  for (int k = 0; k < clscnt4; ++k) {
    output.production[k] = h_production(k);
    output.loss[k] = h_loss(k);
  }
  output.outcome = h_outcome(0);
  if (solver == Solver::Legacy) {
    const auto h_result =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), legacy_result);
    output.legacy_result = h_result(0);
  }
  return output;
}

// Test-only reference assembled from the documented chemistry equations,
// without using production linmat, indprd, or the ordered-solve helper.
std::array<Real, clscnt4>
dense_backward_euler_reference(const SolverCase &input) {
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
    rhs[row] =
        static_cast<long double>(input.state[row + 1]) + dt * source[row];
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
  case 0:
    return input.rates[1];
  case 2:
    return f[0];
  case 4:
    return f[8];
  case 5:
    return f[1];
  case 12:
    return f[5];
  case 13:
    return f[2];
  case 17:
    return f[6];
  case 26:
    return f[3];
  case 27:
    return f[4];
  case 29:
    return f[7];
  default:
    return 0;
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
                 Real(1e-4), Real(2e-4),  Real(3e-4)};
  input.forcing = {Real(1e-9),  Real(1e-11), Real(2e-11),
                   Real(1e-11), Real(1e-11), Real(1e4),
                   Real(2e4),   Real(3e4),   Real(5e-11)};
  return input;
}

void require_atomic_failure(const SolverCase &input,
                            const ImpSolOutcome expected) {
  const auto output = run_on_device(input);
  REQUIRE(output.outcome == expected);
  for (int j = 0; j < gas_pcnst; ++j) {
    if (std::isnan(input.state[j])) {
      REQUIRE(std::isnan(output.state[j]));
    } else {
      REQUIRE(std::memcmp(&output.state[j], &input.state[j], sizeof(Real)) ==
              0);
    }
  }
  for (int k = 0; k < clscnt4; ++k) {
    REQUIRE(output.production[k] == 0);
    REQUIRE(output.loss[k] == 0);
  }
}

// Artificial numerical fixtures in the low-level solver's work-entry units.
// Gas entries 0-5 are mol/mol dry air; aerosol mass and modal number entries
// retain their legacy work-array basis (no physical-unit conversion here).
// rates[0,2:6] and het are s^-1; rates[1] is the completed H2O2 source in
// mol/mol/s. forcing is already normalized: destination-entry units per second,
// NOT the unnormalized molecules/cm^3/s accepted by gas_phase_chemistry.
constexpr int source_destination[extcnt] = {3, 6, 14, 27, 28, 13, 18, 30, 5};

bool is_number_entry(const int j) {
  return j == 13 || j == 18 || j == 26 || j == 30;
}

Real comparison_initial_value(const int j) {
  return is_number_entry(j) ? Real(1e8) * Real(j + 1)
                            : Real(1e-9) * Real(j + 1);
}

enum class ComparisonKind { Identity, Reaction, Source, Removal, Coupled };

// EAMxx default coupling intervals: day/ATM_NCPL for ne30 (48), ne256
// (144), ne1024 (864), from driver-mct/cime_config/config_component_e3sm.xml.
// MAM uses that process dt, outside the subcycled cloud-physics subgroup.
// These are fixed fixture constants; case overrides may change model dt.
constexpr std::array<Real, 3> comparison_timesteps = {Real(1800), Real(600),
                                                      Real(100)};

struct ComparisonCase {
  std::string name;
  ComparisonKind kind;
  int channel = -1;
  SolverCase input;

  ComparisonCase(const std::string &name_in, const ComparisonKind kind_in,
                 const int channel_in = -1)
      : name(name_in), kind(kind_in), channel(channel_in) {
    input.state[0] = Real(2e-8);
  }
};

std::vector<ComparisonCase> comparison_cases() {
  std::vector<ComparisonCase> cases;
  ComparisonCase identity("identity", ComparisonKind::Identity);
  for (int j = 1; j < gas_pcnst; ++j) {
    identity.input.state[j] = comparison_initial_value(j);
  }
  cases.push_back(identity);

  const char *reaction_names[rxntot] = {
      "R0_H2O2_photolysis", "R1_H2O2_source", "R2_H2O2_OH_loss",
      "R3_SO2_to_H2SO4",    "R4_DMS_to_SO2",  "R5_DMS_half_SO2_yield",
      "R6_DMS_to_SO2"};
  for (int k = 0; k < rxntot; ++k) {
    ComparisonCase c(reaction_names[k], ComparisonKind::Reaction, k);
    // Isolated first-order channels have tau=1000 s; R1 is a source.
    c.input.rates[k] = k == 1 ? Real(1e-12) : Real(1e-3);
    if (k == 0 || k == 2) {
      c.input.state[1] = Real(1e-9);
    } else if (k == 3) {
      c.input.state[3] = Real(1e-9);
    } else if (k >= 4) {
      c.input.state[4] = Real(1e-9);
    }
    cases.push_back(c);
  }

  for (int k = 0; k < extcnt; ++k) {
    const int j = source_destination[k];
    ComparisonCase c("external_source_" + std::to_string(k) + "_entry_" +
                         std::to_string(j),
                     ComparisonKind::Source, k);
    c.input.forcing[k] = is_number_entry(j) ? Real(1000) : Real(1e-12);
    cases.push_back(c);
  }
  // Every solved entry, including all gases, aerosol masses, and four modal
  // numbers, gets its own tau=1000 s heterogeneous-removal case.
  for (int j = 1; j < gas_pcnst; ++j) {
    ComparisonCase c("heterogeneous_removal_entry_" + std::to_string(j),
                     ComparisonKind::Removal, j);
    c.input.state[j] = comparison_initial_value(j);
    c.input.het[j] = Real(1e-3);
    cases.push_back(c);
  }

  struct Regime {
    const char *name;
    double dms, so2, sink; // s^-1
  };
  // Chemical loss timescales tau_DMS/tau_SO2 (seconds):
  // weak: 1e6/5e5; moderate: 1e3/1e5; fast DMS: 1/1e5;
  // fast SO2: 1e5/1. The H2SO4 sink has tau=1e6 s throughout.
  // These span k*dt << 1, order unity, and up to 1800, including both
  // orderings of the chain's fast and slow species. They are numerical
  // stress cases, not regional or seasonal atmospheric states.
  const Regime regimes[] = {{"weak", 1e-6, 2e-6, 1e-6},
                            {"moderate", 1e-3, 1e-5, 1e-6},
                            {"strong_fast_dms", 1, 1e-5, 1e-6},
                            {"strong_fast_so2", 1e-5, 1, 1e-6}};
  for (const bool combined : {false, true}) {
    for (const auto &regime : regimes) {
      for (const std::string pattern :
           {"mixed", "zero_chain_products", "contrasting_species"}) {
        ComparisonCase c(std::string(combined ? "combined/" : "coupled/") +
                             regime.name + "/" + pattern,
                         ComparisonKind::Coupled);
        auto &input = c.input;
        for (int j = 1; j < gas_pcnst; ++j) {
          input.state[j] = comparison_initial_value(j);
        }
        input.rates[3] = Real(regime.so2);
        input.rates[4] = Real(regime.dms);
        input.het[2] = Real(regime.sink);
        if (pattern == "zero_chain_products") {
          input.state[2] = input.state[3] = 0;
        } else if (pattern == "contrasting_species") {
          input.state[4] = Real(1e-12);
          input.state[3] = Real(1e-6);
          input.state[2] = Real(1e-15);
        }
        if (combined) {
          // Other sinks have tau=3.33e5 to 1e6 s, SO2=2e6 s, DMS=4e6 s.
          for (int j = 1; j < gas_pcnst; ++j) {
            input.het[j] = Real(1e-6) * Real(1 + j % 3);
          }
          input.het[2] = Real(regime.sink);
          input.het[3] = Real(5e-7);
          input.het[4] = Real(2.5e-7);
          // H2O2 chemical loss shares tau_DMS; its source is independent.
          input.rates[0] = Real(0.4) * Real(regime.dms);
          input.rates[2] = Real(0.6) * Real(regime.dms);
          input.rates[1] = Real(5e-12);
          // DMS loss splits 1/4, 1/2, 1/4; R5 retains half its sulfur as SO2.
          input.rates[4] = Real(0.25) * Real(regime.dms);
          input.rates[5] = Real(0.5) * Real(regime.dms);
          input.rates[6] = Real(0.25) * Real(regime.dms);
          for (int k = 0; k < extcnt; ++k) {
            // A fixed source scale per 1e6 s, independent of timestep/pattern.
            input.forcing[k] =
                comparison_initial_value(source_destination[k]) * Real(1e-6);
          }
        }
        cases.push_back(c);
      }
    }
  }
  return cases;
}

// Independent isolated-channel backward-Euler formulas. No mechanism helper
// or production matrix is used. Unrelated entries retain their initial value.
std::array<Real, clscnt4>
isolated_backward_euler_reference(const ComparisonCase &c) {
  const auto &input = c.input;
  std::array<Real, gas_pcnst> expected = input.state;
  const long double dt = input.dt;
  const int k = c.channel;
  if (c.kind == ComparisonKind::Reaction) {
    const long double rate = input.rates[k];
    if (k == 1) {
      expected[1] = Real(input.state[1] + dt * rate);
    } else {
      const int reactant = (k == 0 || k == 2) ? 1 : (k == 3 ? 3 : 4);
      const long double remaining = input.state[reactant] / (1 + dt * rate);
      expected[reactant] = Real(remaining);
      if (k >= 3) {
        const int product = k == 3 ? 2 : 3;
        const long double yield = k == 5 ? 0.5L : 1.0L;
        expected[product] =
            Real(input.state[product] + yield * dt * rate * remaining);
      }
    }
  } else if (c.kind == ComparisonKind::Source) {
    const int j = source_destination[k];
    expected[j] = Real(input.state[j] + dt * input.forcing[k]);
  } else if (c.kind == ComparisonKind::Removal) {
    expected[k] = Real(input.state[k] / (1 + dt * input.het[k]));
  }
  std::array<Real, clscnt4> answer;
  std::copy(expected.begin() + 1, expected.end(), answer.begin());
  return answer;
}

// Isolated equations have closed-form solutions; coupled cases use an
// independently assembled backward-Euler system. Neither uses solver output
// or the production mechanism helpers to construct the expected endpoint.
std::array<Real, clscnt4>
known_backward_euler_solution(const ComparisonCase &c) {
  return c.kind == ComparisonKind::Coupled
             ? dense_backward_euler_reference(c.input)
             : isolated_backward_euler_reference(c);
}

// An inactive equation has neither loss nor source/coupling. A steady state
// with balanced nonzero production/loss is active and can round differently.
bool comparison_entry_is_inactive(const SolverCase &input, const int j) {
  if (input.het[j] != 0 || source_for_entry(input, j - 1) != 0) {
    return false;
  }
  const auto &r = input.rates;
  switch (j) {
  case 1:
    return r[0] == 0 && r[2] == 0;
  case 2:
    return r[3] == 0;
  case 3:
    return r[3] == 0 && r[4] == 0 && r[5] == 0 && r[6] == 0;
  case 4:
    return r[4] == 0 && r[5] == 0 && r[6] == 0;
  default:
    return true;
  }
}

// Float Newton corrections have a roundoff floor, including for identity.
// Keep 1e-12 in double and use eight epsilons as a float portability margin.
// This is test-only; production legacy::rel_err remains 1e-3.
Real comparison_tolerance() {
  return std::max(Real(1e-12), Real(8) * std::numeric_limits<Real>::epsilon());
}

} // namespace

TEST_CASE("direct gas chemistry matches independent backward Euler",
          "[gas_chem]") {
  const auto input = nominal_case();
  const auto output = run_on_device(input);
  const auto reference = dense_backward_euler_reference(input);
  REQUIRE(output.outcome == ImpSolOutcome::Converged);
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
      expected_loss +=
          (input.rates[4] + input.rates[5] + input.rates[6]) * output.state[j];
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

TEST_CASE(
    "direct gas chemistry preserves zero rates and handles timestep scales",
    "[gas_chem]") {
  SolverCase zero;
  zero.state = nominal_case().state;
  const auto unchanged = run_on_device(zero);
  REQUIRE(unchanged.outcome == ImpSolOutcome::Converged);
  for (int j = 0; j < gas_pcnst; ++j) {
    REQUIRE(unchanged.state[j] == zero.state[j]);
  }

  for (const Real dt : {Real(1e-8), Real(1e6)}) {
    auto input = nominal_case();
    input.dt = dt;
    const auto output = run_on_device(input);
    const auto reference = dense_backward_euler_reference(input);
    REQUIRE(output.outcome == ImpSolOutcome::Converged);
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
  REQUIRE(output.outcome == ImpSolOutcome::Converged);
  REQUIRE(output.state[3] == Approx(1e-6));
  REQUIRE(output.state[5] == Approx(1e-6));
  REQUIRE(std::memcmp(&output.state[0], &input.state[0], sizeof(Real)) == 0);

  input = SolverCase{};
  input.state[5] = 3e-6; // SOAG
  input.forcing[8] = -1e-6;
  const auto signed_source_output = run_on_device(input);
  REQUIRE(signed_source_output.outcome == ImpSolOutcome::Converged);
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

TEST_CASE("direct gas chemistry matches known backward Euler solutions",
          "[gas_chem][known_solution]") {
  auto c = GENERATE(Catch::Generators::from_range(comparison_cases()));
  const Real dt = GENERATE(Catch::Generators::from_range(comparison_timesteps));
  c.input.dt = dt;
  const auto &input = c.input;
  const auto expected = known_backward_euler_solution(c);
  const auto direct = run_on_device(input, Solver::Direct);
  const Real tolerance = comparison_tolerance();
  CAPTURE(c.name, dt);
  INFO("known-solution tolerance = " << std::scientific << tolerance);
  REQUIRE(direct.outcome == ImpSolOutcome::Converged);
  REQUIRE(direct.state[0] == input.state[0]);

  // This test runs independently of the legacy solver and its convergence.
  // Scale only by input and known solution, keeping gas, aerosol-mass, and
  // modal-number errors dimensionless without depending on either solver.
  double squared_error = 0;
  for (int j = 1; j < gas_pcnst; ++j) {
    const Real expected_value = expected[j - 1];
    CAPTURE(j);
    INFO(std::scientific << "initial = " << input.state[j] << ", expected = "
                         << expected_value << ", direct = " << direct.state[j]);
    REQUIRE(std::isfinite(expected_value));
    REQUIRE(std::isfinite(direct.state[j]));
    REQUIRE(expected_value >= 0);
    REQUIRE(direct.state[j] >= 0);
    const double scale = std::max(std::abs(double(input.state[j])),
                                  std::abs(double(expected_value)));
    if (scale == 0) {
      REQUIRE(direct.state[j] == 0);
    } else {
      const double error = (double(direct.state[j]) - expected_value) / scale;
      CAPTURE(scale, error);
      REQUIRE(std::abs(error) <= tolerance);
      squared_error += error * error;
    }
    if (comparison_entry_is_inactive(input, j)) {
      REQUIRE(direct.state[j] == input.state[j]);
    }
  }
  const double l2_error = std::sqrt(squared_error);
  const double l2_limit = std::sqrt(double(clscnt4)) * tolerance;
  CAPTURE(l2_error, l2_limit);
  REQUIRE(l2_error <= l2_limit);
}

TEST_CASE("legacy and direct gas chemistry agree for deterministic full steps",
          "[gas_chem][comparison]") {
  auto c = GENERATE(Catch::Generators::from_range(comparison_cases()));
  const Real dt = GENERATE(Catch::Generators::from_range(comparison_timesteps));
  c.input.dt = dt; // Rates and initial conditions stay fixed across timesteps.
  const auto &input = c.input;
  const Real tolerance = comparison_tolerance();
  CAPTURE(c.name, dt);
  INFO("comparison tolerance = " << std::scientific << tolerance);

  // Separate allocations/copies in each invocation prevent state aliasing.
  const auto reference = run_on_device(input, Solver::Legacy, tolerance);
  const auto direct = run_on_device(input, Solver::Direct);
  const auto &result = reference.legacy_result;
  CAPTURE(result.outcome, result.failed_attempts, result.cut_count,
          result.accepted_steps, result.requested_interval,
          result.accepted_interval);
  // A retry/substep solves a different discrete problem, so stop before
  // interpreting endpoint differences as direct-solver accuracy failures.
  REQUIRE(result.success());
  REQUIRE(reference.outcome == ImpSolOutcome::Converged);
  REQUIRE(result.failed_attempts == 0);
  REQUIRE(result.cut_count == 0);
  REQUIRE(result.accepted_steps == 1);
  REQUIRE(result.requested_interval == dt);
  REQUIRE(result.accepted_interval == dt);
  REQUIRE(direct.outcome == ImpSolOutcome::Converged);
  REQUIRE(reference.state[0] == input.state[0]);
  REQUIRE(direct.state[0] == input.state[0]);

  const auto expected = known_backward_euler_solution(c);
  // Dimensionless L2 = sqrt(sum_j ((direct_j - legacy_j)/scale_j)^2),
  // scale_j = max(|initial_j|, |legacy_j|, |independent BE_j|). Scaling
  // each entry prevents the large modal numbers from hiding gas/mass errors.
  const double l2_limit = std::sqrt(double(clscnt4)) * tolerance;
  double squared_error = 0;
  for (int j = 1; j < gas_pcnst; ++j) {
    const Real expected_value = expected[j - 1];
    CAPTURE(j);
    INFO(std::scientific << "initial = " << input.state[j]
                         << ", expected = " << expected_value
                         << ", legacy = " << reference.state[j]
                         << ", direct = " << direct.state[j]);
    REQUIRE(std::isfinite(expected_value));
    REQUIRE(std::isfinite(reference.state[j]));
    REQUIRE(std::isfinite(direct.state[j]));
    REQUIRE(expected_value >= 0);
    REQUIRE(reference.state[j] >= 0);
    REQUIRE(direct.state[j] >= 0);
    const double scale = std::max({std::abs(double(input.state[j])),
                                   std::abs(double(reference.state[j])),
                                   std::abs(double(expected_value))});
    if (scale == 0) {
      // No arbitrary absolute floor: an unsourced zero must remain zero.
      REQUIRE(reference.state[j] == 0);
      REQUIRE(direct.state[j] == 0);
    } else {
      const double paired_error =
          (double(direct.state[j]) - reference.state[j]) / scale;
      const double direct_error =
          (double(direct.state[j]) - expected_value) / scale;
      const double legacy_error =
          (double(reference.state[j]) - expected_value) / scale;
      CAPTURE(scale, paired_error, direct_error, legacy_error);
      REQUIRE(std::abs(paired_error) <= tolerance);
      REQUIRE(std::abs(direct_error) <= tolerance);
      REQUIRE(std::abs(legacy_error) <= tolerance);
      squared_error += paired_error * paired_error;
    }
    // Direct preserves inactive entries exactly; Newton's identity update
    // may introduce roundoff in the legacy path.
    if (comparison_entry_is_inactive(input, j)) {
      REQUIRE(direct.state[j] == input.state[j]);
      const double roundoff_limit =
          8 * double(std::numeric_limits<Real>::epsilon()) *
          std::abs(double(input.state[j]));
      REQUIRE(std::abs(double(reference.state[j]) - input.state[j]) <=
              roundoff_limit);
    }
  }
  const double l2_error = std::sqrt(squared_error);
  CAPTURE(l2_error, l2_limit);
  REQUIRE(l2_error <= l2_limit);
}
