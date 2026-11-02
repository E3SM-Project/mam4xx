// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_HPP
#define MAM4XX_HPP

// This header makes all MAM4 processes available.

#include "aero_config.hpp"
#include "aero_model.hpp"
#include "aero_model_emissions.hpp"
#include "aero_rad_props.hpp"
#include "aging.hpp"
#include "calcsize.hpp"
#include "coagulation.hpp"
#include "compute_o3_column_density.hpp"
#include "convproc.hpp"
#include "diagnostic_arrays.hpp"
#include "drydep.hpp"
#include "gas_chem.hpp"
#include "gas_chem_mechanism.hpp"
#include "gas_phase_chemistry.hpp"
#include "gasaerexch.hpp"
#include "hetfrz.hpp"
#include "lin_strat_chem.hpp"
#include "mam4_amicphys.hpp"
#include "mam4_types.hpp"
#include "mo_drydep.hpp"
#include "mo_gas_phase_chemdr.hpp"
#include "mo_photo.hpp"
#include "mo_setext.hpp"
#include "mo_sethet.hpp"
#include "mo_setinv.hpp"
#include "mo_setsox.hpp"
#include "modal_aero_calcsize.hpp"
#include "modal_aero_opt.hpp"
#include "ndrop.hpp"
#include "nucleate_ice.hpp"
#include "nucleation.hpp"
#include "rename.hpp"
#include "spitfire_transport.hpp"
#include "tropopause.hpp"
#include "vertical_interpolation.hpp"
#include "water_uptake.hpp"
#include "wet_dep.hpp"

namespace mam4 {

// Returns mam4xx's version string.
const char *version();

// Returns mam4xx's git revision hash, or "unknown" if not found.
const char *revision();

// Returns true iff this build has changes that weren't committed to the git
// repo.
bool has_uncommitted_changes();

} // namespace mam4

#endif
