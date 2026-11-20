#ifndef MAM4XX_DROPMIXNUC_HPP
#define MAM4XX_DROPMIXNUC_HPP

#include <mam4xx/aero_modes.hpp>
#include <mam4xx/atmosphere.hpp>
#include <mam4xx/diagnostic_arrays.hpp>
#include <mam4xx/mam4_amicphys.hpp>
#include <mam4xx/mam4_constants.hpp>
#include <mam4xx/mo_drydep.hpp>
#include <mam4xx/mo_setsox.hpp>
#include <mam4xx/ndrop.hpp>

#include <ekat_team_policy_utils.hpp>

namespace mam4::aerosol_cloud_interaction {

using Real              = mam4::Real;
using View1D            = mam4::DeviceType::view_1d<Real>;
using View2D            = mam4::DeviceType::view_2d<Real>;
using View3D            = mam4::DeviceType::view_3d<Real>;
using ThreadTeam        = mam4::ThreadTeam;
using TeamPolicyFactory = ekat::TeamPolicyFactory<ekat::KokkosTypes<ekat::DefaultDevice>::ExeSpace>;

//------------------------------------------
// Aerosol-Cloud Interaction  configuration
//------------------------------------------

enum class CriticalRelation {
  DiluteKappaKohler,
};

struct DropletActivationConfig {
  bool             enable;
  CriticalRelation critical_relation;
};

struct HeterogeneousFreezingConfig {
  bool enable;
};

struct IceNucleationConfig {
  bool enable;
  Real subgrid;
  Real so4_size_threshold;
};

struct DropMixNucConfig {
  DropletActivationConfig droplet_activation;
  IceNucleationConfig     ice_nucleation;
};

struct Config {
  // NOTE: In the greatness of time, we'd like things organized this way...
  //DropletActivationConfig     droplet_activation;
  //HeterogeneousFreezingConfig heterogeneous_freezing;
  //IceNucleationConfig         ice_nucleation;

  // NOTE: ... for now, we articulate things in terms of dropmixnuc
  DropMixNucConfig dropmixnuc;
};

//-------------
// Work arrays
//-------------

// WorkArrays collects all the work arrays used by the DropMixNuc class. These work arrays are
// used as needed by DropMixNuc but ownership is not transferred. This allows DropMixNuc to
// share memory with other processes.
struct WorkArrays {
  // single-column variable work arrays [nlev]
  std::array<ColumnView, 8> columns;
  // constituent work arrays [nlev, gas_pcnst]
  std::array<View2D, 3> constituents;
  // fixed-species work arrays [nlev, nfs]
  std::array<View2D, 1> fixed_species;
};

// DropMixNuc advances the prognostic aerosol state in place by subjecting it to various
// aerosol microphysics processes (determined by the configuration).
class DropMixNuc final {
public:
  // Creates a DropMixNuc instance with internally-managed work arrays.
  explicit DropMixNuc(const DropMixNucConfig &config);
  // Creates a DropMixNuc instance with externally-managed work arrays.
  DropMixNuc(const DropMixNucConfig &config, WorkArrays &work_arrays);
  ~DropMixNuc();

  // No casual creating, copying, or moving.
  DropMixNuc() = delete;
  DropMixNuc(const DropMixNuc&) = delete;
  DropMixNuc(const DropMixNuc&&) = delete;

  // Runs configured aerosol microphysics processes over a single timestep, updating the given set
  // of (single column) prognostics under the given prevailing atmosphere conditions.
  void step(const mam4::Atmosphere &atmosphere,
            mam4::Prognostics &prognostics,
            Real dt);

private:
  DropMixNucConfig config_;
  struct { // work arrays
    bool owned; // true iff the zero-argument DropMixNuc constructor was used
  } work_;

  void set_work_arrays(const WorkArrays &work_arrays, bool assume_ownership);
};

} // namespace mam4::aerosol_cloud_interaction

#endif
