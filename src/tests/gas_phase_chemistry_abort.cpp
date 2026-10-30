// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <mam4xx/mam4.hpp>
#include <string>

int main(int argc, char **argv) {
#if defined(MAM4XX_GAS_CHEMISTRY_SOLVER_LEGACY) || defined(NDEBUG)
  // The wrapper returns diagnostics in these configurations instead of
  // aborting.
  return 77;
#else
  if (argc != 2 || (std::string(argv[1]) != "invalid_interval" &&
                    std::string(argv[1]) != "negative_endpoint")) {
    return 2;
  }
  Kokkos::initialize(argc, argv);
  using mam4::Real;
  using namespace mam4::gas_chemistry;
  Real state[gas_pcnst];
  for (int j = 0; j < gas_pcnst; ++j) {
    state[j] = Real(j + 1) * Real(1e-9);
  }
  const bool invalid_interval = std::string(argv[1]) == "invalid_interval";
  if (!invalid_interval) {
    state[5] = Real(1e-6);
  }
  // Each invocation has its own runtime; no initialized process is forked.
  Real photo_rates[mam4::mo_photo::phtcnt] = {};
  Real forcing[extcnt] = {};
  Real invariants[nfs] = {};
  Real heterogeneous_rates[gas_pcnst] = {};
  // Synthetic M=1 and zero oxidants/photolysis isolate the external SOAG
  // source.
  invariants[indexm] = 1;
  forcing[8] = invalid_interval ? Real(0) : Real(-2e-6);
  mam4::microphysics::gas_phase_chemistry(
      Real(300), invalid_interval ? Real(-1) : Real(1), photo_rates, forcing,
      invariants, heterogeneous_rates, state);
  Kokkos::finalize();
  std::cerr << "Gas chemistry unexpectedly returned instead of aborting\n";
  return 0;
#endif
}
