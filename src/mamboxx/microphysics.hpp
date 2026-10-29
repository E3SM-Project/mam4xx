#ifndef MAM4XX_MICROPHYSICS_HPP
#define MAM4XX_MICROPHYSICS_HPP

#include <mam4_config.hpp>
#include <mam4xx/aero_config.hpp>
#include <mam4xx/aero_modes.hpp>
#include <mam4xx/atmosphere.hpp>
#include <mam4xx/diagnostic_arrays.hpp>
#include <mam4xx/mam4_amicphys.hpp>
#include <mam4xx/mam4_constants.hpp>
#include <mam4xx/mam4_types.hpp>
#include <mam4xx/mo_drydep.hpp>
#include <mam4xx/mo_setsox.hpp>

#include <ekat_team_policy_utils.hpp>

namespace mam4::microphysics {

using Real              = mam4::Real;
using View1D            = mam4::DeviceType::view_1d<Real>;
using View2D            = mam4::DeviceType::view_2d<Real>;
using View3D            = mam4::DeviceType::view_3d<Real>;
using ThreadTeam        = mam4::ThreadTeam;
using TeamPolicyFactory = ekat::TeamPolicyFactory<ekat::KokkosTypes<ekat::DefaultDevice>::ExeSpace>;


// Aerosol microphysics makes use of several views with specific dimensions. These type aliases
// give a hint about what a particular view contains (and how it's indexed).
using ConstituentsView            = View2D; // [nlev][gas_pcnst] NOTE: sometimes reversed
using FixedSpeciesView            = View2D; // [nlev][nfs]
using SingleLevelConstituentsView = View1D; // [gas_pcnst]
using PhotolysisRatesView         = View2D; // [nlev][mam4::mo_photo::phtcnt]

// Config represents a complete aerosol microphysics configuration, and is bound to a specific
// Microphysics instance.
struct Config {
  bool aqueous_chemistry;
  bool gas_chemistry;
  bool gas_aerosol_exchange;
  bool rename;
  bool nucleation;
  bool coagulation;

  // parameterization-specific configurations
  mam4::mo_setsox::Config setsox;
  AmicPhysConfig          amicphys;

  // FIXME: These could be better organized
  int n_so4_monolayers_pcage;
};

// ColumnLocation stores location-specific information because aerosol physics is strongly coupled
// to solar radiation. Passed to microphysics to advance the state of a particular column.
struct ColumnLocation {
  Real latitude;                    // latitude of column center [radians]
  Real surface_geopotential;        // column geopotential, evaluated at top of atmosphere [m^2/s^2]
  Real snow_height;                 // snow height/depth on land [m]
  Real surface_temperature;         // temperature at earth's surface [K]
  Real surface_pressure;            // atmosphere pressure at earth's surface [Pa]
  Real precip_liquid_surface_mass;  // total liquid rain mass at earth's surface [kg]
  Real precip_ice_surface_mass;     // total ice rain mass at earth's surface [kg]

  // land use fraction by land type [-]
  std::array<Real, mam4::mo_drydep::n_land_type> land_use_fraction;
  // seasons indexed by land type
  std::array<int, mam4::mo_drydep::n_land_type> season_by_land_type;
};

// ExternalConditions stores data describing tendencies and forcings from non-aerosol sources.
struct ExternalConditions {
  ConstituentsView forcing; // constituent external forcing [??]
  ColumnView       prain;   // stratiform rain production rate [kg/kg/s]
  ColumnView       nevapr;  // evaporation from stratiform rain [kg/kg/s]
  ColumnView       u_wind;  // u component of wind velocity [m/s]
  ColumnView       v_wind;  // v component of wind velocity [m/s]

  ColumnView       sw_solar_flux_dn;   // downwelling short-wave solar flux at earth's surface (interface) [W/m2]

  SingleLevelConstituentsView constituent_fluxes; // fluxes of gas and aerosol species [kg/m^2/s]

  // present only when config.enable_extra_diagnostics is set
  ColumnView       gas_dry_deposition_flux;   // [kg/m^2/s?]
  ConstituentsView gas_chemistry_dvmrdt;      // [??], NOTE: layout: [gas_pcnst][nlev]
  ConstituentsView aqueous_chemistry_dvmrdt;  // [??], NOTE: layout: [gas_pcnst][nlev]
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
  // photolysis rates [nlev, mam4::mo_photo::phtcnt]
  PhotolysisRatesView photolysis_rates;
  // sethet work array (size determined using sethet functions)
  View1D sethet;
};

// Microphysics advances the prognostic aerosol state in place by subjecting it to various
// aerosol microphysics processes (determined by the configuration).
class Microphysics final {
public:
  // Creates a Microphysics instance with internally-managed work arrays.
  explicit Microphysics(const Config &config);
  // Creates a Microphysics instance with externally-managed work arrays.
  Microphysics(const Config &config, const WorkArrays &work_arrays);

  ~Microphysics();

  // No casual creating, copying, or moving.
  Microphysics() = delete;
  Microphysics(const Microphysics&) = delete;
  Microphysics(const Microphysics&&) = delete;

  void enable_extra_diagnostics(mam4::MicrophysDiagnosticArrays arrays);

  // Runs configured aerosol microphysics processes over a single timestep, updating the given set
  // of (single column) prognostics under the given prevailing atmosphere conditions.
  void step(const mam4::Atmosphere &atmosphere,
            mam4::Prognostics &prognostics,
            const ColumnLocation &location,
            ExternalConditions &external_conditions,
            Real dt);

  // Given the atmospheric conditions and the prognostic state, compute microphysics diagnostics.
  void compute_diagnostics(const mam4::Atmosphere &atmosphere,
                           const mam4::Prognostics &prognostics,
                           mam4::Diagnostics &diagnostics);

private:
  Config config_;

  struct { // work arrays
    ColumnView cmfdqr; // dq/dt for convection [nlev]

    ConstituentsView het_rates;               // heterogeneous freezing rates
    ConstituentsView qq;                      // interstitial constituents
    ConstituentsView qqcw;                    // cloudborne constituents
    ConstituentsView vmr, vmr0, vmrcw,        // volume mixing ratios
                     vmr_pregas, vmr_precld;

    ConstituentsView dqdt_aqso4;   // so4 aqueous chemistry tendency
    ConstituentsView dqdt_aqh2so4; // h2so4 aqueous chemistry tendency

    FixedSpeciesView invariants; // chemical invariants

    PhotolysisRatesView photolysis_rates;

    View1D sethet; // sethet parameterization work array

    bool owned; // true iff the single-argument Microphysics constructor was used
  } work_;

  struct {
    bool enabled;
    mam4::MicrophysDiagnosticArrays arrays;
  } extra_diagnostics_;

  void set_work_arrays(const WorkArrays &work_arrays, bool assume_ownership);
  void set_external_forcing(const ColumnView &external_forcing);
  void set_invariants(const mam4::Atmosphere &atmosphere);
  //void compute_o3_column_density(const Atmosphere &atmosphere, Prognostics &prognostics);
  //void photo_table(State &state, Real dt);
  void sethet(const Atmosphere &atmosphere,
              const ColumnLocation &location,
              const ExternalConditions &external_conditions,
              Real dt);
  void drydep_xactive(const Atmosphere &atmosphere, const ColumnLocation &location,
                      ExternalConditions &external_conditions);
  void gas_phase_chemistry(const Atmosphere &atmosphere, const ExternalConditions &external_conditions, Real dt);
  void setsox(const Atmosphere &atmosphere, const ExternalConditions &external_conditions, Real dt);
  void modal_aero_amicphys_intr(const Atmosphere &atmosphere, Real dt);
  void vmr2mmr_cw(State &state, Real dt);
  //void linoz(State &state, Real dt);
};

} // namespace mam4::microphysics

#endif
