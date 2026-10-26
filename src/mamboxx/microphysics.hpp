#ifndef MAM4XX_MICROPHYSICS_HPP
#define MAM4XX_MICROPHYSICS_HPP

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

// This type represents a complete aerosol microphysics configuration.
struct Config {
  bool gas_chemistry;
  bool gas_aerosol_exchange;
  bool rename;
  bool nucleation;
  bool coagulation;
};

class Microphysics final {
  Microphysics(const Config &config);
  ~Microphysics() = default;

  // Dear C++ compiler: no casual creating, copying, or moving.
  Microphysics() = delete;
  Microphysics(const Microphysics&) = delete;
  Microphysics(const Microphysics&&) = delete;

  // Runs configured aerosol microphysics processes over a single timestep, updating the given set
  // of (single column) prognostics under the given prevailing atmosphere conditions.
  void step(const mam4::Atmosphere &atmosphere,
            mam4::Prognostics &prognostics,
            const ExternalForcing &external_forcing,
            Invariants &invariants,
            Real dt);

  // Given the atmospheric conditions and the prognostic state, compute microphysics diagnostics.
  void compute_diagnostics(const mam4::Atmosphere &atmosphere,
                           const mam4::Prognostics &prognostics,
                           mam4::Diagnostics &diagnostics);

private:

  Config config_;
  View2D qq_, qqcw_; // work arrays
};


} // namespace mam4::microphysics

#endif
