#ifndef MAM4XX_MICROPHYSICS_HPP
#define MAM4XX_MICROPHYSICS_HPP

#include "mam4xx/gas_chem_mechanism.hpp"
#include <mam4_config.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/atmosphere.hpp>
#include <mam4xx/mam4_constants.hpp>
#include <mam4xx/mam4_types.hpp>

#include <ekat_team_policy_utils.hpp>

namespace mam4::microphysics {

using Real              = mam4::Real;
using View1D            = mam4::DeviceType::view_1d<Real>;
using View2D            = mam4::DeviceType::view_2d<Real>;
using View3D            = mam4::DeviceType::view_3d<Real>;
using ThreadTeam        = mam4::ThreadTeam;
using TeamPolicyFactory = ekat::TeamPolicyFactory<ekat::KokkosTypes<ekat::DefaultDevice>::ExeSpace>;

// This is a type alias for device-resident column work array data.
using ColumnWorkArray = View2D;

// This is a type alias for device-resident constituent external forcing data.
// Layout: [constituent][sector][level]
using ExternalForcing = View3D;

// This is a type alias for device-resident invariant data.
// Layout: [level][invariant-index]
using Invariants = View2D;

// Config represents a complete aerosol microphysics configuration, and is bound to a specific
// Microphysics instance.
struct Config {
  bool gas_chemistry;
  bool gas_aerosol_exchange;
  bool rename;
  bool nucleation;
  bool coagulation;
};

// ColumnLocation stores location-specific information because aerosol physics is strongly coupled
// to solar radiation. Passed to microphysics to advance the state of a particular column.
struct ColumnLocation {
  Real latitude; // latitude of column center in radians
};

// WorkArrays collects all the work arrays used by the Microphysics class. These work arrays are
// used as needed by Microphysics but ownership is not transferred. This allows Microphysics to
// share memory with other processes.
struct WorkArrays {
  // single-column variable work arrays [nlev]
  std::array<ColumnView, 8> columns;
  // constituent work arrays [nlev, gas_pcnst]
  std::array<View2D, 3> constituents;
  // fixed-species work arrays [nlev, nfs]
  std::array<View2D, 1> fixed_species;
};

// Microphysics advances the prognostic aerosol state in place by subjecting it to various
// aerosol microphysics processes (determined by the configuration).
class Microphysics final {

  Microphysics(const Config &config, const WorkArrays &work_arrays);
  ~Microphysics() = default;

  // No casual creating, copying, or moving.
  Microphysics() = delete;
  Microphysics(const Microphysics&) = delete;
  Microphysics(const Microphysics&&) = delete;

  // Runs configured aerosol microphysics processes over a single timestep, updating the given set
  // of (single column) prognostics under the given prevailing atmosphere conditions.
  void step(const mam4::Atmosphere &atmosphere,
            mam4::Prognostics &prognostics,
            const ColumnLocation &location,
            const ExternalForcing &external_forcing,
            Real dt);

  // Given the atmospheric conditions and the prognostic state, compute microphysics diagnostics.
  void compute_diagnostics(const mam4::Atmosphere &atmosphere,
                           const mam4::Prognostics &prognostics,
                           mam4::Diagnostics &diagnostics);

private:
  Config config_;

  // work arrays
  View2D qq_, qqcw_, het_rates_;
  View2D invariants_;

  void set_external_forcing(const ColumnView &external_forcing);
  void set_invariants(const mam4::Atmosphere &atmosphere, Invariants &invariants);
  //void compute_o3_column_density(const Atmosphere &atmosphere, Prognostics &prognostics);
  //void photo_table(State &state, Real dt);
  void sethet(const Atmosphere &state, const ColumnLocation &location,
              Real surface_geopotential, Real dt);
  void drydep_xactive(State &state, Real dt);
  void gas_phase_chemistry(State &state, Real dt);
  void setsox(State &state, Real dt);
  void modal_aero_amicphys_intr(State &state, Real dt);
  void vmr2mmr_cw(State &state, Real dt);
  //void linoz(State &state, Real dt);
};


} // namespace mam4::microphysics

#endif
