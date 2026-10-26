#include "microphysics.hpp"

#include <mam4xx/mo_setinv.hpp>

namespace mam4::microphysics {

namespace {

void set_external_forcing(State &state, Real dt) {
  // TODO: add when appropriate
}

void set_invariants(const mam4::Atmosphere &atmosphere,
                    Invariants &invariants) {
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);
  View1D cnst_offline[mam4::mo_setinv::num_tracer_cnst];
  // FIXME: populate offline constituents 
  Kokkos::parallel_for("mamboxx::set_invariants", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    mam4::mo_setinv::setinv(team,                              // in
                            invariants,                        // out
                            atmosphere.temperature,            // in
                            atmosphere.vapor_mixing_ratio,     // in
                            cnst_offline,                      // in
                            atmosphere.pressure);              // in
  });
}

void compute_o3_column_density(const Atmosphere &atmosphere,
                               Prognostics &prognostics,
                               
}

void photo_table(State &state, Real dt) {
}

void sethet(State &state, Real dt) {
}

void drydep_xactive(State &state, Real dt) {
}

void gas_phase_chemistry(State &state, Real dt) {
}

void setsox(State &state, Real dt) {
}

void modal_aero_amicphys_intr(State &state, Real dt) {
}

void vmr2mmr_cw(State &state, Real dt) {
}

void linoz(State &state, Real dt) {
}

} // anonymous namespace

// FIXME: initialize work arrays here!
Microphysics::Microphysics(const Config &config) {
}

// Runs configured aerosol microphysics processes over a single timestep, updating the given set of
// (single column) prognostics under the given prevailing atmosphere conditions.
void Microphysics::step(const mam4::Atmosphere &atmosphere,
                        mam4::Prognostics &prognostics,
          const ExternalForcing &external_forcing,
          Invariants &invariants,
          Real dt) {
  size_t n_lev = atmosphere.temperature.size();
 
  // For guidance here, see MAMMicrophysics::run_impl in EAMxx.

  // TODO: add chlorine loading for linoz
  // TODO: add elevated emissions
  // TODO: add insolation based on orbital parameters/date and location (or just directly)
 
  // MAMMicrophysics::run_microphysics_kernels
  set_invariants(atmosphere, invariants);
  compute_o3_column_density(atmosphere, dt);
  if constexpr (mam4::nlev > 1) {
    photo_table(state, dt);
    sethet(state, dt);
  }
  drydep_xactive(state, dt);
  gas_phase_chemistry(state, dt);
  setsox(state, dt);
  modal_aero_amicphys_intr(state, dt);
  vmr2mmr_cw(state, dt);
  linoz(state, dt);

}

void compute_diagnostics(const mam4::Atmosphere &atmosphere,
                         const mam4::Prognostics &prognostics,
                         mam4::Diagnostics &diagnostics) {
  diagnostics = {};
}

} // namespace mam4::microphysics
