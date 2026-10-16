// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/convproc.hpp>
#include <validation.hpp>

#include <iostream>

// This driver computes the binary or ternary nucleation rate for the given
// input.

void usage() {
  std::cerr << "convproc_driver: a Skywalker driver for validating the "
               "MAM4 convproc parameterizations."
            << std::endl;
  std::cerr << "convproc_driver: usage:" << std::endl;
  std::cerr << "convproc_driver <input.yaml>" << std::endl;
  exit(0);
}

using namespace skywalker;

// Parameterizations used by the convproc process.
void update_tendency_final(Ensemble *ensemble);
void compute_column_tendency(Ensemble *ensemble);
void ma_resuspend_convproc(Ensemble *ensemble);
void ma_precpevap(Ensemble *ensemble);
void ma_precpprod(Ensemble *ensemble);
void ma_precpevap_convproc(Ensemble *ensemble);
void initialize_dcondt(Ensemble *ensemble);
void compute_downdraft_mixing_ratio(Ensemble *ensemble);
void aer_vol_num_hygro(Ensemble *ensemble);
void compute_wup(Ensemble *ensemble);
void compute_massflux(Ensemble *ensemble);
void compute_ent_det_dp(Ensemble *ensemble);
void compute_midlev_height(Ensemble *ensemble);
void initialize_tmr_array(Ensemble *ensemble);
void update_qnew_ptend(Ensemble *ensemble);
void compute_wetdep_tend(Ensemble *ensemble);
void ma_activate_convproc(Ensemble *ensemble);
void compute_activation_tend(Ensemble *ensemble);
void compute_updraft_mixing_ratio(Ensemble *ensemble);
void ma_convproc_tend(Ensemble *ensemble);
void ma_convproc_dp_intr(Ensemble *ensemble);
void ma_convproc_intr(Ensemble *ensemble);
void compute_tendencies(Ensemble *ensemble);

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
  EKAT_REQUIRE_MSG(settings.has("function"),
                   "No function specified in mam4xx.settings!");

  // Dispatch to the requested function.
  auto func_name = settings.get("function");
  try {
    if (func_name == "update_tendency_final") {
      update_tendency_final(ensemble);
    } else if (func_name == "compute_column_tendency") {
      compute_column_tendency(ensemble);
    } else if (func_name == "ma_resuspend_convproc") {
      ma_resuspend_convproc(ensemble);
    } else if (func_name == "ma_precpevap") {
      ma_precpevap(ensemble);
    } else if (func_name == "ma_precpprod") {
      ma_precpprod(ensemble);
    } else if (func_name == "ma_precpevap_convproc") {
      ma_precpevap_convproc(ensemble);
    } else if (func_name == "initialize_dcondt") {
      initialize_dcondt(ensemble);
    } else if (func_name == "compute_downdraft_mixing_ratio") {
      compute_downdraft_mixing_ratio(ensemble);
    } else if (func_name == "aer_vol_num_hygro") {
      aer_vol_num_hygro(ensemble);
    } else if (func_name == "compute_wup") {
      compute_wup(ensemble);
    } else if (func_name == "compute_massflux") {
      compute_massflux(ensemble);
    } else if (func_name == "compute_ent_det_dp") {
      compute_ent_det_dp(ensemble);
    } else if (func_name == "compute_midlev_height") {
      compute_midlev_height(ensemble);
    } else if (func_name == "initialize_tmr_array") {
      initialize_tmr_array(ensemble);
    } else if (func_name == "update_qnew_ptend") {
      update_qnew_ptend(ensemble);
    } else if (func_name == "compute_wetdep_tend") {
      compute_wetdep_tend(ensemble);
    } else if (func_name == "ma_activate_convproc") {
      ma_activate_convproc(ensemble);
    } else if (func_name == "compute_activation_tend") {
      compute_activation_tend(ensemble);
    } else if (func_name == "compute_updraft_mixing_ratio") {
      compute_updraft_mixing_ratio(ensemble);
    } else if (func_name == "ma_convproc_tend") {
      ma_convproc_tend(ensemble);
    } else if (func_name == "ma_convproc_dp_intr") {
      ma_convproc_dp_intr(ensemble);
    } else if (func_name == "convective_processing") {
      ma_convproc_intr(ensemble);
    } else if (func_name == "compute_tendencies") {
      compute_tendencies(ensemble);
    } else {
      std::cerr << "Error: Test name not recognized:" << func_name << std::endl;
      exit(1);
    }
  } catch (std::exception &e) {
    std::cerr << argv[0] << ": Error: " << e.what() << std::endl;
    throw;
  }

  // Write out a Python module.
  std::cout << argv[0] << ": writing " << output_file << std::endl;
  ensemble->write(output_file);

  // Clean up.
  mam4::validation::finalize(ensemble);
}
