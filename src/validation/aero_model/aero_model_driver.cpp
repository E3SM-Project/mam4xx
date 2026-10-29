// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/aero_model.hpp>

#include <iostream>
#include <validation.hpp>

// This driver computes the binary or ternary rename for the given
// input.

void usage() {
  std::cerr << "aero_model_driver: a Skywalker driver for validating the "
               "MAM4 rename parameterizations."
            << std::endl;
  std::cerr << "aero_model_driver: usage:" << std::endl;
  std::cerr << "aero_model_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;

// Parameterizations used by the aero_model process.
void modal_aero_bcscavcoef_init(Ensemble *ensemble);
void calc_1_impact_rate(Ensemble *ensemble);
void modal_aero_bcscavcoef_get(Ensemble *ensemble);
void aero_model_wetdep(Ensemble *ensemble);
void aero_model_wetdep_convproc(Ensemble *ensemble);
void aero_model_calcsize_water_uptake_dr(Ensemble *ensemble);

int main(int argc, char **argv) {
  if (argc == 1) {
    usage();
  }
  mam4::validation::initialize(argc, argv, mam4::validation::default_fpes);
  std::string input_file = argv[1];
  std::string output_file = mam4::validation::output_name(input_file);
  std::cout << argv[0] << ": reading " << input_file << std::endl;

  // Load the ensemble. Any error encountered is fatal.
  Ensemble *ensemble = skywalker::load_ensemble(input_file, "mam4xx");

  // the settings.
  Settings settings = ensemble->settings();
  if (!settings.has("function")) {
    std::cerr << "No function specified in mam4xx.settings!" << std::endl;
    exit(1);
  }

  // Dispatch to the requested function.
  auto func_name = settings.get("function");
  try {
    if (func_name == "modal_aero_bcscavcoef_init") {
      modal_aero_bcscavcoef_init(ensemble);
    } else if (func_name == "calc_1_impact_rate") {
      calc_1_impact_rate(ensemble);
    } else if (func_name == "modal_aero_bcscavcoef_get") {
      modal_aero_bcscavcoef_get(ensemble);
    } else if (func_name == "aero_model_wetdep") {
      aero_model_wetdep(ensemble);
    } else if (func_name == "aero_model_wetdep_convproc" ||
               func_name == "aero_model_wetdep_convproc_no_gas_no_aerosol") {
      aero_model_wetdep_convproc(ensemble);
    } else if (func_name == "aero_model_calcsize_water_uptake_dr") {
      aero_model_calcsize_water_uptake_dr(ensemble);
    } else {
      std::cerr << "Error: Function name '" << func_name
                << "' does not have an implemented test!" << std::endl;
      exit(1);
    }

  } catch (std::exception &e) {
    std::cerr << argv[0] << ": Error: " << e.what() << std::endl;
  }

  // Write out a Python module.
  std::cout << argv[0] << ": writing " << output_file << std::endl;
  ensemble->write(output_file);

  // Clean up.
  mam4::validation::finalize(ensemble);
}
