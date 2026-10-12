#include "mam4xx/mo_setinv.hpp"
#include "mamboxx.hpp"
#include <mam4xx/mam4.hpp>

#include <ekat_assert.hpp>
#include <ekat_team_policy_utils.hpp>

// Unresolved questions/issues:
// 1. How do we accommodate different numbers of vertical levels? mam4::nlev
//    is currently a compile-time parameter.
// 2. We need an AtmosphereState instead of an immutable AtmosphereConfig, in
//    which water vapor/ice/etc are tracked and evolved. This means additional
//    input data in the YAML file. See the DryAtmosphere struct in mam_coupling.hpp
//    for the essentials.
// 3. How do we handle emissions and external forcing? Use input from the YAML file?

namespace mamboxx {

namespace {

using View1D            = mam4::DeviceType::view_1d<Real>;
using View2D            = mam4::DeviceType::view_2d<Real>;
using ThreadTeam        = mam4::ThreadTeam;
using TeamPolicyFactory = ekat::TeamPolicyFactory<ekat::KokkosTypes<ekat::DefaultDevice>::ExeSpace>;

struct State {
  AtmosphereConfig atmosphere;
  AerosolState     aerosols;
  GasState         gases;

  View2D invariants;
  int n_lev, k_top, k_bottom;
};

void initialize(const Config &config, State &state) {
  state.n_lev = config.atmosphere.temperature.size();
  EKAT_REQUIRE(state.n_lev <= mam4::nlev,
      "requested number of vertical levels (" << state.n_lev <<
      ") exceeds compile-time limit (" << mam4::nlev <<
      ") -- reconfigure with MAM4XX_NUM_VERTICALS set to a higher limit");
  state.k_top = 0;
  state.k_bottom = state.n_lev;
}

void external_forcing(State &state, Real dt) {
  // TODO: add when appropriate
}

void set_invariants(State &state, Real dt) {
  auto policy = TeamPolicyFactory::get_default_team_policy(1, state.n_lev);
  View1D cnst_offline[mam4::mo_setinv::num_tracer_cnst];
  // FIXME: populate offline constituents 
  Kokkos::parallel_for("mamboxx::set_invariants", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    mam4::mo_setinv::setinv(team,                                    // in
                            state.invariants,                        // out
                            state.atmosphere.temperature,            // in
                            state.atmosphere.vapor_mixing_ratio,     // in
                            cnst_offline,                            // in
                            state.atmosphere.pressure);              // in
  });
}

void compute_o3_column_density(State &state, Real dt) {
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


void step(State &state, Real dt) {
  size_t n_lev = state.atmosphere.temperature.size();
 
  // For guidance here, see MAMMicrophysics::run_impl in EAMxx.

  // TODO: add chlorine loading for linoz
  // TODO: add elevated emissions
  // TODO: add insolation based on orbital parameters/date and location (or just directly)
 
  // MAMMicrophysics::run_microphysics_kernels
  external_forcing(state, dt);
  set_invariants(state, dt);
  compute_o3_column_density(state, dt);
  photo_table(state, dt);
  sethet(state, dt);
  drydep_xactive(state, dt);
  gas_phase_chemistry(state, dt);
  setsox(state, dt);
  modal_aero_amicphys_intr(state, dt);
  vmr2mmr_cw(state, dt);
  linoz(state, dt);

}

void update_diagnostics(const State &state, Real dt, Diagnostics &diagnostics) {
  diagnostics = {};
}

} // anonymous namespace

Result run(const Config &config) {
  State state = {};
  Diagnostics diagnostics = {};

  initialize(config, state);

  for (int i = 0; i < config.time.nstep; ++i) {
    step(state, config.time.dt);
    update_diagnostics(state, config.time.dt, diagnostics);
  }

  Result result = {};
  result.success = true;
  result.aerosols = state.aerosols;
  result.gases = state.gases;
  return result;
}

}
