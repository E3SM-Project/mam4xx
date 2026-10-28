#include "microphysics.hpp"

#include <mam4xx/mo_setinv.hpp>

namespace mam4::microphysics {

// FIXME: initialize work arrays here!
Microphysics::Microphysics(const Config &config, const WorkArrays &work_arrays):
  config_(config), qq_(work_arrays.constituents[0]), qqcw_(work_arrays.constituents[1]),
  het_rates_(work_arrays.constituents[2]), invariants_(work_arrays.fixed_species[0])
{
}

// Runs configured aerosol microphysics processes over a single timestep, updating the given set of
// (single column) prognostics under the given prevailing atmosphere conditions.
void Microphysics::step(const mam4::Atmosphere &atmosphere,
                        mam4::Prognostics &prognostics,
                        const ColumnLocation &location,
                        const ExternalForcing &external_forcing,
                        Real dt) {
  size_t n_lev = atmosphere.temperature.size();
 
  // For guidance here, see MAMMicrophysics::run_impl in EAMxx.

  // TODO: add chlorine loading for linoz
  // TODO: add elevated emissions
  // TODO: add insolation based on orbital parameters/date and location (or just directly)
 
  // MAMMicrophysics::run_microphysics_kernels
  set_invariants(atmosphere, invariants_);
  //compute_o3_column_density(atmosphere, dt);
  if constexpr (mam4::nlev > 1) {
    //photo_table(state, dt);
    sethet(atmosphere, location, dt);
  }
  drydep_xactive(state, dt);
  gas_phase_chemistry(state, dt);
  setsox(state, dt);
  modal_aero_amicphys_intr(state, dt);
  vmr2mmr_cw(state, dt);
  linoz(state, dt);

}

void Microphysics::compute_diagnostics(const mam4::Atmosphere &atmosphere,
                                       const mam4::Prognostics &prognostics,
                                       mam4::Diagnostics &diagnostics) {
  diagnostics = {};
}

void Microphysics::set_external_forcing(const ColumnView &external_forcing) {
  // TODO: add when appropriate
}

void Microphysics::set_invariants(const mam4::Atmosphere &atmosphere,
                                  Invariants &invariants) {
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);
  View1D cnst_offline[mam4::mo_setinv::num_tracer_cnst];
  // FIXME: populate offline constituents 
  Kokkos::parallel_for("mam4::Microphysics::set_invariants", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    mam4::mo_setinv::setinv(team,                              // in
                            invariants,                        // out
                            atmosphere.temperature,            // in
                            atmosphere.vapor_mixing_ratio,     // in
                            cnst_offline,                      // in
                            atmosphere.pressure);              // in
  });
}

/*
void Microphysics::compute_o3_column_density(const Atmosphere &atmosphere,
                                             Prognostics &prognostics,
}

void Microphysics::photo_table(State &state, Real dt) {
}
*/

void Microphysics::sethet(const Atmosphere &atmosphere, const ColumnLocation &location,
                          Real surface_geopotential, Real dt) {
  // set sethet
  Kokkos::parallel_for("mam4::Microphysics::sethet", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
      const auto phis= dry_atm.phis(icol);
      const Real col_lat = col_latitudes(icol);  // column latitude (degrees?)
                                                 // convert column latitude to radians
      const auto prain_icol        = ekat::subview(prain, icol);
      const auto nevapr_icol       = ekat::subview(nevapr, icol);
      const auto invariants_icol = ekat::subview(invariants, icol);
      const auto vmr_icol = ekat::subview(vmr, icol);

      const auto work_set_het_icol = ekat::subview(work_set_het, icol);
      auto work_set_het_ptr = (Real *)work_set_het_icol.data();
      const int sethet_work_len = mam4::mo_sethet::get_work_len_sethet();
      const auto work_sethet_call = view_1d(work_set_het_ptr, sethet_work_len);
      work_set_het_ptr += sethet_work_len;
      mam4::mo_sethet::sethet(team, atmosphere, het_rates_,
          location.latitude, surface_geopotential, cmfdqr, prain_icol,
          nevapr_icol, dt, invariants_, vmr_,
          work_sethet_call);
      });

}

void Microphysics::drydep_xactive(State &state, Real dt) {
}

void Microphysics::gas_phase_chemistry(State &state, Real dt) {
}

void Microphysics::setsox(State &state, Real dt) {
}

void Microphysics::modal_aero_amicphys_intr(State &state, Real dt) {
}

void Microphysics::vmr2mmr_cw(State &state, Real dt) {
}

/*
void Microphysics::linoz(State &state, Real dt) {
}
*/

} // namespace mam4::microphysics
