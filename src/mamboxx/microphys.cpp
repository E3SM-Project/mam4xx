#include "microphys.hpp"

namespace mamboxx {

void Microphys::initialize(const MicrophysConfig &config) {
  this->config = config;
}

void Microphys::step(Real dt) {
  // See run_microphysics_kernels() in microphys_functions.cpp for details.

  // set external forcing
  // set invariants
  // extract state
  // compute_o3_column_density
  // photo_table
  // sethet
  // drydep_xactive
  // gas_phase_chemistry
  // setsox
  // modal_aero_amicphys_intr
  // linoz
  // update state
  // diagnostics
}

void Microphys::finalize() {
}

} // namespace mamboxx
