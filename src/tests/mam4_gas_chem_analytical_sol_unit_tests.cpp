// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/floating_point.hpp>
#include <mam4xx/gas_chem.hpp>
#include <mam4xx/mam4_types.hpp>

#include <catch2/catch.hpp>
#include <cmath>
#include <vector>

using mam4::DeviceType;
using mam4::Real;
using namespace mam4::gas_chemistry;

namespace {

struct GasChemInputs {
  Real base_sol[gas_pcnst];
  Real reaction_rates[rxntot];
  Real het_rates[gas_pcnst];
  Real extfrc[extcnt];
  Real delt;
};

struct GasChemOutputs {
  Real base_sol[gas_pcnst];
  Real prod[clscnt4];
  Real loss[clscnt4];
  int outcome;
  int accepted_steps;
  int non_converged_species_idx;
  int non_converged_species_count;
};

void setup_realistic_inputs(GasChemInputs &inputs, const Real delt) {
  inputs.delt = delt;

  for (int i = 0; i < gas_pcnst; ++i) {
    inputs.base_sol[i] = 1.0e-15;
  }

  // Representative mixing ratios [vmr]
  inputs.base_sol[0] = 5.0e-8;  // O3
  inputs.base_sol[1] = 1.0e-9;  // H2O2
  inputs.base_sol[2] = 1.0e-12; // H2SO4
  inputs.base_sol[3] = 2.0e-10; // SO2
  inputs.base_sol[4] = 5.0e-11; // DMS
  inputs.base_sol[5] = 1.0e-11; // SOAG

  // Aerosol species
  for (int i = 6; i < gas_pcnst; ++i) {
    inputs.base_sol[i] = 1.0e-12 * (1.0 + 0.1 * (i - 6));
  }

  // Reaction rates [1/s]
  inputs.reaction_rates[0] = 1.0e-5;  // J_H2O2
  inputs.reaction_rates[1] = 1.0e-16; // HO2 + HO2 -> H2O2
  inputs.reaction_rates[2] = 1.7e-6;  // OH + H2O2
  inputs.reaction_rates[3] = 1.0e-6;  // OH + SO2 -> H2SO4
  inputs.reaction_rates[4] = 5.0e-6;  // OH + DMS (abstraction)
  inputs.reaction_rates[5] = 2.0e-6;  // OH + DMS (addition)
  inputs.reaction_rates[6] = 1.0e-7;  // NO3 + DMS

  // Heterogeneous loss/washout rates [1/s]
  for (int i = 0; i < gas_pcnst; ++i) {
    inputs.het_rates[i] = 1.0e-5;
  }
  inputs.het_rates[2] = 5.0e-4; // H2SO4 fast condensation sink

  // In-situ external forcings [vmr/s]
  for (int i = 0; i < extcnt; ++i) {
    inputs.extfrc[i] = 0.0;
  }
  inputs.extfrc[0] = 1.0e-15; // SO2 emissions
  inputs.extfrc[1] = 2.0e-16; // Aerosol emissions
}

void dispatch_solvers_to_device(const GasChemInputs &inputs,
                                GasChemOutputs &out_imp,
                                GasChemOutputs &out_ana) {
  typename DeviceType::view_1d<Real> in_base("in_base", gas_pcnst);
  typename DeviceType::view_1d<Real> in_rxt("in_rxt", rxntot);
  typename DeviceType::view_1d<Real> in_het("in_het", gas_pcnst);
  typename DeviceType::view_1d<Real> in_ext("in_ext", extcnt);

  // Copy inputs to device
  auto h_base = Kokkos::create_mirror_view(in_base);
  for (int i = 0; i < gas_pcnst; ++i)
    h_base(i) = inputs.base_sol[i];
  Kokkos::deep_copy(in_base, h_base);

  auto h_rxt = Kokkos::create_mirror_view(in_rxt);
  for (int i = 0; i < rxntot; ++i)
    h_rxt(i) = inputs.reaction_rates[i];
  Kokkos::deep_copy(in_rxt, h_rxt);

  auto h_het = Kokkos::create_mirror_view(in_het);
  for (int i = 0; i < gas_pcnst; ++i)
    h_het(i) = inputs.het_rates[i];
  Kokkos::deep_copy(in_het, h_het);

  auto h_ext = Kokkos::create_mirror_view(in_ext);
  for (int i = 0; i < extcnt; ++i)
    h_ext(i) = inputs.extfrc[i];
  Kokkos::deep_copy(in_ext, h_ext);

  // Device outputs for imp_sol
  typename DeviceType::view_1d<Real> d_base_imp("d_base_imp", gas_pcnst);
  typename DeviceType::view_1d<Real> d_prod_imp("d_prod_imp", clscnt4);
  typename DeviceType::view_1d<Real> d_loss_imp("d_loss_imp", clscnt4);
  typename DeviceType::view_1d<int> d_meta_imp("d_meta_imp", 4);

  // Device outputs for analytical_sol
  typename DeviceType::view_1d<Real> d_base_ana("d_base_ana", gas_pcnst);
  typename DeviceType::view_1d<Real> d_prod_ana("d_prod_ana", clscnt4);
  typename DeviceType::view_1d<Real> d_loss_ana("d_loss_ana", clscnt4);
  typename DeviceType::view_1d<int> d_meta_ana("d_meta_ana", 4);

  const Real delt = inputs.delt;

  Kokkos::parallel_for(
      "run_gas_chem_solvers_on_device", 1, KOKKOS_LAMBDA(const int) {
        // --- 1. Run imp_sol on Device ---
        Real base_imp[gas_pcnst];
        for (int i = 0; i < gas_pcnst; ++i)
          base_imp[i] = in_base(i);
        Real rxt[rxntot];
        for (int i = 0; i < rxntot; ++i)
          rxt[i] = in_rxt(i);
        Real het[gas_pcnst];
        for (int i = 0; i < gas_pcnst; ++i)
          het[i] = in_het(i);
        Real ext[extcnt];
        for (int i = 0; i < extcnt; ++i)
          ext[i] = in_ext(i);

        Real prod_imp[clscnt4] = {};
        Real loss_imp[clscnt4] = {};
        Real epsilon[clscnt4] = {};
        imp_slv_inti(epsilon);
        bool factor[itermax];
        for (int i = 0; i < itermax; ++i)
          factor[i] = true;
        ImpSolResult res_imp;

        imp_sol(base_imp, rxt, het, ext, delt, factor, epsilon, prod_imp,
                loss_imp, res_imp);

        for (int i = 0; i < gas_pcnst; ++i)
          d_base_imp(i) = base_imp[i];
        for (int i = 0; i < clscnt4; ++i) {
          d_prod_imp(i) = prod_imp[i];
          d_loss_imp(i) = loss_imp[i];
        }
        d_meta_imp(0) = static_cast<int>(res_imp.outcome);
        d_meta_imp(1) = res_imp.accepted_steps;
        d_meta_imp(2) = res_imp.non_converged_species_idx;
        d_meta_imp(3) = res_imp.non_converged_species_count;

        // --- 2. Run analytical_sol on Device ---
        Real base_ana[gas_pcnst];
        for (int i = 0; i < gas_pcnst; ++i)
          base_ana[i] = in_base(i);
        Real prod_ana[clscnt4] = {};
        Real loss_ana[clscnt4] = {};
        ImpSolResult res_ana;

        analytical_sol(base_ana, rxt, het, ext, delt, prod_ana, loss_ana,
                       res_ana);

        for (int i = 0; i < gas_pcnst; ++i)
          d_base_ana(i) = base_ana[i];
        for (int i = 0; i < clscnt4; ++i) {
          d_prod_ana(i) = prod_ana[i];
          d_loss_ana(i) = loss_ana[i];
        }
        d_meta_ana(0) = static_cast<int>(res_ana.outcome);
        d_meta_ana(1) = res_ana.accepted_steps;
        d_meta_ana(2) = res_ana.non_converged_species_idx;
        d_meta_ana(3) = res_ana.non_converged_species_count;
      });

  Kokkos::fence();

  // Copy outputs back to host
  auto h_base_imp = Kokkos::create_mirror_view(d_base_imp);
  Kokkos::deep_copy(h_base_imp, d_base_imp);
  auto h_prod_imp = Kokkos::create_mirror_view(d_prod_imp);
  Kokkos::deep_copy(h_prod_imp, d_prod_imp);
  auto h_loss_imp = Kokkos::create_mirror_view(d_loss_imp);
  Kokkos::deep_copy(h_loss_imp, d_loss_imp);
  auto h_meta_imp = Kokkos::create_mirror_view(d_meta_imp);
  Kokkos::deep_copy(h_meta_imp, d_meta_imp);

  for (int i = 0; i < gas_pcnst; ++i)
    out_imp.base_sol[i] = h_base_imp(i);
  for (int i = 0; i < clscnt4; ++i) {
    out_imp.prod[i] = h_prod_imp(i);
    out_imp.loss[i] = h_loss_imp(i);
  }
  out_imp.outcome = h_meta_imp(0);
  out_imp.accepted_steps = h_meta_imp(1);
  out_imp.non_converged_species_idx = h_meta_imp(2);
  out_imp.non_converged_species_count = h_meta_imp(3);

  auto h_base_ana = Kokkos::create_mirror_view(d_base_ana);
  Kokkos::deep_copy(h_base_ana, d_base_ana);
  auto h_prod_ana = Kokkos::create_mirror_view(d_prod_ana);
  Kokkos::deep_copy(h_prod_ana, d_prod_ana);
  auto h_loss_ana = Kokkos::create_mirror_view(d_loss_ana);
  Kokkos::deep_copy(h_loss_ana, d_loss_ana);
  auto h_meta_ana = Kokkos::create_mirror_view(d_meta_ana);
  Kokkos::deep_copy(h_meta_ana, d_meta_ana);

  for (int i = 0; i < gas_pcnst; ++i)
    out_ana.base_sol[i] = h_base_ana(i);
  for (int i = 0; i < clscnt4; ++i) {
    out_ana.prod[i] = h_prod_ana(i);
    out_ana.loss[i] = h_loss_ana(i);
  }
  out_ana.outcome = h_meta_ana(0);
  out_ana.accepted_steps = h_meta_ana(1);
  out_ana.non_converged_species_idx = h_meta_ana(2);
  out_ana.non_converged_species_count = h_meta_ana(3);
}

void dispatch_analytical_to_device(const GasChemInputs &inputs,
                                   GasChemOutputs &out_ana) {
  typename DeviceType::view_1d<Real> in_base("in_base_a", gas_pcnst);
  typename DeviceType::view_1d<Real> in_rxt("in_rxt_a", rxntot);
  typename DeviceType::view_1d<Real> in_het("in_het_a", gas_pcnst);
  typename DeviceType::view_1d<Real> in_ext("in_ext_a", extcnt);

  auto h_base = Kokkos::create_mirror_view(in_base);
  for (int i = 0; i < gas_pcnst; ++i)
    h_base(i) = inputs.base_sol[i];
  Kokkos::deep_copy(in_base, h_base);

  auto h_rxt = Kokkos::create_mirror_view(in_rxt);
  for (int i = 0; i < rxntot; ++i)
    h_rxt(i) = inputs.reaction_rates[i];
  Kokkos::deep_copy(in_rxt, h_rxt);

  auto h_het = Kokkos::create_mirror_view(in_het);
  for (int i = 0; i < gas_pcnst; ++i)
    h_het(i) = inputs.het_rates[i];
  Kokkos::deep_copy(in_het, h_het);

  auto h_ext = Kokkos::create_mirror_view(in_ext);
  for (int i = 0; i < extcnt; ++i)
    h_ext(i) = inputs.extfrc[i];
  Kokkos::deep_copy(in_ext, h_ext);

  typename DeviceType::view_1d<Real> d_base_ana("d_base_a_out", gas_pcnst);
  typename DeviceType::view_1d<Real> d_prod_ana("d_prod_a_out", clscnt4);
  typename DeviceType::view_1d<Real> d_loss_ana("d_loss_a_out", clscnt4);
  typename DeviceType::view_1d<int> d_meta_ana("d_meta_a_out", 4);

  const Real delt = inputs.delt;

  Kokkos::parallel_for(
      "run_analytical_sol_on_device", 1, KOKKOS_LAMBDA(const int) {
        Real base_ana[gas_pcnst];
        for (int i = 0; i < gas_pcnst; ++i)
          base_ana[i] = in_base(i);
        Real rxt[rxntot];
        for (int i = 0; i < rxntot; ++i)
          rxt[i] = in_rxt(i);
        Real het[gas_pcnst];
        for (int i = 0; i < gas_pcnst; ++i)
          het[i] = in_het(i);
        Real ext[extcnt];
        for (int i = 0; i < extcnt; ++i)
          ext[i] = in_ext(i);

        Real prod_ana[clscnt4] = {};
        Real loss_ana[clscnt4] = {};
        ImpSolResult res_ana;

        analytical_sol(base_ana, rxt, het, ext, delt, prod_ana, loss_ana,
                       res_ana);

        for (int i = 0; i < gas_pcnst; ++i)
          d_base_ana(i) = base_ana[i];
        for (int i = 0; i < clscnt4; ++i) {
          d_prod_ana(i) = prod_ana[i];
          d_loss_ana(i) = loss_ana[i];
        }
        d_meta_ana(0) = static_cast<int>(res_ana.outcome);
        d_meta_ana(1) = res_ana.accepted_steps;
        d_meta_ana(2) = res_ana.non_converged_species_idx;
        d_meta_ana(3) = res_ana.non_converged_species_count;
      });

  Kokkos::fence();

  auto h_base_ana = Kokkos::create_mirror_view(d_base_ana);
  Kokkos::deep_copy(h_base_ana, d_base_ana);
  auto h_prod_ana = Kokkos::create_mirror_view(d_prod_ana);
  Kokkos::deep_copy(h_prod_ana, d_prod_ana);
  auto h_loss_ana = Kokkos::create_mirror_view(d_loss_ana);
  Kokkos::deep_copy(h_loss_ana, d_loss_ana);
  auto h_meta_ana = Kokkos::create_mirror_view(d_meta_ana);
  Kokkos::deep_copy(h_meta_ana, d_meta_ana);

  for (int i = 0; i < gas_pcnst; ++i)
    out_ana.base_sol[i] = h_base_ana(i);
  for (int i = 0; i < clscnt4; ++i) {
    out_ana.prod[i] = h_prod_ana(i);
    out_ana.loss[i] = h_loss_ana(i);
  }
  out_ana.outcome = h_meta_ana(0);
  out_ana.accepted_steps = h_meta_ana(1);
  out_ana.non_converged_species_idx = h_meta_ana(2);
  out_ana.non_converged_species_count = h_meta_ana(3);
}

} // namespace

TEST_CASE("analytical_sol vs imp_sol on Device: Small timestep convergence",
          "[gas_chem]") {
  GasChemInputs inputs;
  setup_realistic_inputs(inputs, 0.5); // dt = 0.5 s

  GasChemOutputs out_imp, out_ana;
  dispatch_solvers_to_device(inputs, out_imp, out_ana);

  REQUIRE(out_imp.outcome == static_cast<int>(ImpSolOutcome::Converged));
  REQUIRE(out_ana.outcome == static_cast<int>(ImpSolOutcome::Converged));

  const Real tol = 1.0e-8;

  for (int spc = 1; spc <= 4; ++spc) {
    const Real rel_diff =
        std::abs(out_imp.base_sol[spc] - out_ana.base_sol[spc]) /
        std::abs(out_ana.base_sol[spc]);
    REQUIRE(rel_diff < tol);
  }

  for (int spc = 5; spc < gas_pcnst; ++spc) {
    const Real rel_diff =
        std::abs(out_imp.base_sol[spc] - out_ana.base_sol[spc]) /
        std::abs(out_ana.base_sol[spc]);
    REQUIRE(rel_diff < tol);
  }
}

TEST_CASE(
    "analytical_sol vs imp_sol on Device: Atmospheric timestep (1800s)",
    "[gas_chem]") {
  GasChemInputs inputs;
  setup_realistic_inputs(inputs, 1800.0); // dt = 1800 s

  GasChemOutputs out_imp, out_ana;
  dispatch_solvers_to_device(inputs, out_imp, out_ana);

  REQUIRE(out_ana.outcome == static_cast<int>(ImpSolOutcome::Converged));
  REQUIRE(out_ana.accepted_steps == 1);

  // Both solvers must ensure physical non-negativity and finiteness on device
  for (int i = 0; i < gas_pcnst; ++i) {
    REQUIRE(out_ana.base_sol[i] >= 0.0);
    REQUIRE(out_imp.base_sol[i] >= 0.0);
    REQUIRE(std::isfinite(out_ana.base_sol[i]));
    REQUIRE(std::isfinite(out_imp.base_sol[i]));
  }

  // Consistent qualitative physical behavior
  REQUIRE(out_ana.base_sol[4] < inputs.base_sol[4]); // DMS loss
  REQUIRE(out_imp.base_sol[4] < inputs.base_sol[4]);

  for (int k = 0; k < clscnt4; ++k) {
    REQUIRE(out_ana.prod[k] >= 0.0);
    REQUIRE(out_ana.loss[k] >= 0.0);
    REQUIRE(std::isfinite(out_ana.prod[k]));
    REQUIRE(std::isfinite(out_ana.loss[k]));
  }
}

TEST_CASE("analytical_sol on Device: Pure exponential decay exactness",
          "[gas_chem]") {
  GasChemInputs inputs;
  for (int i = 0; i < gas_pcnst; ++i) {
    inputs.base_sol[i] = 1.0e-10;
    inputs.het_rates[i] = 2.0e-4; // 1/s
  }
  for (int i = 0; i < rxntot; ++i)
    inputs.reaction_rates[i] = 0.0;
  for (int i = 0; i < extcnt; ++i)
    inputs.extfrc[i] = 0.0;
  inputs.delt = 1000.0;

  GasChemOutputs out_ana;
  dispatch_analytical_to_device(inputs, out_ana);

  REQUIRE(out_ana.outcome == static_cast<int>(ImpSolOutcome::Converged));

  const Real expected = 1.0e-10 * std::exp(-2.0e-4 * 1000.0);
  for (int i = 1; i < gas_pcnst; ++i) {
    const Real rel_err =
        std::abs(out_ana.base_sol[i] - expected) / expected;
    REQUIRE(rel_err < 1.0e-12);
  }
}

TEST_CASE(
    "analytical_sol on Device: Degenerate loss rates in sulfur chain",
    "[gas_chem]") {
  GasChemInputs inputs;
  setup_realistic_inputs(inputs, 600.0);

  // Equal loss rates between SO2 and DMS
  const Real equal_lambda = 1.0e-4;
  inputs.reaction_rates[3] = 0.0;
  inputs.reaction_rates[4] = 0.0;
  inputs.reaction_rates[5] = 0.0;
  inputs.reaction_rates[6] = 0.0;
  inputs.het_rates[3] = equal_lambda; // SO2
  inputs.het_rates[4] = equal_lambda; // DMS
  inputs.het_rates[2] = equal_lambda; // H2SO4

  GasChemOutputs out_ana;
  dispatch_analytical_to_device(inputs, out_ana);

  REQUIRE(out_ana.outcome == static_cast<int>(ImpSolOutcome::Converged));
  REQUIRE(std::isfinite(out_ana.base_sol[2]));
  REQUIRE(std::isfinite(out_ana.base_sol[3]));
  REQUIRE(std::isfinite(out_ana.base_sol[4]));
  REQUIRE(out_ana.base_sol[2] >= 0.0);
  REQUIRE(out_ana.base_sol[3] >= 0.0);
  REQUIRE(out_ana.base_sol[4] >= 0.0);
}

TEST_CASE("analytical_sol on Device: Invalid input handling", "[gas_chem]") {
  GasChemInputs inputs;
  setup_realistic_inputs(inputs, -10.0); // Negative dt

  GasChemOutputs out_ana;
  dispatch_analytical_to_device(inputs, out_ana);

  REQUIRE(out_ana.outcome == static_cast<int>(ImpSolOutcome::InvalidInput));

  // Non-finite concentration
  inputs.delt = 10.0;
  inputs.base_sol[1] = NAN;
  dispatch_analytical_to_device(inputs, out_ana);

  REQUIRE(out_ana.outcome == static_cast<int>(ImpSolOutcome::InvalidInput));
}
