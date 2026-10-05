// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_UTILS_HPP
#define MAM4XX_UTILS_HPP

#include "aero_config.hpp"
#include "aero_modes.hpp"
#include "atmosphere.hpp"
#include "gas_chem_mechanism.hpp"
#include "mam4_math.hpp"

#include <ekat_math_utils.hpp>
// This file contains utility-type functions that are available for use by
// various processes, tests, etc.

namespace mam4::utils {

// this function considers 'num' and returns either 'num' (already in bounds) or
// 'high'/'low' if num is outside the bounds
KOKKOS_INLINE_FUNCTION
Real min_max_bound(const Real &low, const Real &high, const Real &num) {
  return mam4::max(low, mam4::min(high, num));
}

// number of constituents in gas chemistry "work arrays"
KOKKOS_INLINE_FUNCTION
constexpr int gas_pcnst() {
  constexpr int gas_pcnst_ = mam4::gas_chemistry::gas_pcnst;
  return gas_pcnst_;
}
// start index of gasses in state_q array of e3sm
KOKKOS_INLINE_FUNCTION
constexpr int gasses_start_ind() {
  return 9;
} // gases start at index 9 (index 10 in Fortran version)

// start index of interstitial (or cloudborne) aerosols in state_q (or qqcw)
// array of e3sm
KOKKOS_INLINE_FUNCTION
constexpr int aero_start_ind() {
  return 15;
} // aerosols start at index 15 (index 16 in Fortran version)

// Because CUDA C++ doesn't allow us to declare and use constants outside of
// KOKKOS_INLINE_FUNCTIONS, we define this macro that allows us to (re)define
// these constants where needed within two such functions so we don't define
// them inconsistently. Yes, it's the 21st century and we're still struggling
// with these basic things.
#define DECLARE_PROG_TRANSFER_CONSTANTS                                        \
  /* mapping of constituent indices to aerosol modes */                        \
  const auto Accum = mam4::ModeIndex::Accumulation;                            \
  const auto Aitken = mam4::ModeIndex::Aitken;                                 \
  const auto Coarse = mam4::ModeIndex::Coarse;                                 \
  const auto PC = mam4::ModeIndex::PrimaryCarbon;                              \
  const auto NoMode = mam4::ModeIndex::None;                                   \
  static const mam4::ModeIndex mode_for_cnst[gas_pcnst()] = {                  \
      NoMode, NoMode, NoMode, NoMode, NoMode, NoMode, /* gases (not aerosols)  \
                                                       */                      \
      Accum,  Accum,  Accum,  Accum,  Accum,  Accum,  Accum,                   \
      Accum,                                  /* 7 aero species + NMR */       \
      Aitken, Aitken, Aitken, Aitken, Aitken, /* 4 aero species + NMR */       \
      Coarse, Coarse, Coarse, Coarse, Coarse, Coarse, Coarse,                  \
      Coarse,                     /* 7 aero species + NMR */                   \
      PC,     PC,     PC,     PC, /* 3 aero species + NMR */                   \
  };                                                                           \
  /* mapping of constituent indices to aerosol species */                      \
  const auto SOA = mam4::AeroId::SOA;                                          \
  const auto SO4 = mam4::AeroId::SO4;                                          \
  const auto POM = mam4::AeroId::POM;                                          \
  const auto BC = mam4::AeroId::BC;                                            \
  const auto NaCl = mam4::AeroId::NaCl;                                        \
  const auto DST = mam4::AeroId::DST;                                          \
  const auto MOM = mam4::AeroId::MOM;                                          \
  const auto NoAero = mam4::AeroId::None;                                      \
  static const mam4::AeroId aero_for_cnst[gas_pcnst()] = {                     \
      NoAero, NoAero, NoAero, NoAero, NoAero, NoAero, /* gases (not aerosols)  \
                                                       */                      \
      SO4,    POM,    SOA,    BC,     DST,    NaCl,   MOM,                     \
      NoAero,                                 /* accumulation mode */          \
      SO4,    SOA,    NaCl,   MOM,    NoAero, /* aitken mode */                \
      DST,    NaCl,   SO4,    BC,     POM,    SOA,    MOM,                     \
      NoAero,                         /* coarse mode */                        \
      POM,    BC,     MOM,    NoAero, /* primary carbon mode */                \
  };                                                                           \
  /* mapping of constituent indices to gases */                                \
  const auto O3 = mam4::GasId::O3;                                             \
  const auto H2O2 = mam4::GasId::H2O2;                                         \
  const auto H2SO4 = mam4::GasId::H2SO4;                                       \
  const auto SO2 = mam4::GasId::SO2;                                           \
  const auto DMS = mam4::GasId::DMS;                                           \
  const auto SOAG = mam4::GasId::SOAG;                                         \
  const auto NoGas = mam4::GasId::None;                                        \
  static const mam4::GasId gas_for_cnst[gas_pcnst()] = {                       \
      O3,    H2O2,  H2SO4, SO2,   DMS,   SOAG,  NoGas, NoGas,                  \
      NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas,                  \
      NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas,                  \
      NoGas, NoGas, NoGas, NoGas, NoGas, NoGas, NoGas,                         \
  };

// Given a Prognostics object, transfers data for interstitial aerosols to the
// chemistry work array q, and cloudborne aerosols to the chemistry work array
// qqcw, both at vertical level k. The input and output quantities are stored as
// number/mass mixing ratios.
// NOTE: this mapping is chemistry-mechanism-specific (see mo_sim_dat.F90
// NOTE: in the relevant preprocessed chemical mechanism)
// NOTE: see mam4xx/aero_modes.hpp to interpret these mode/aerosol/gas
// NOTE: indices
KOKKOS_INLINE_FUNCTION
void transfer_prognostics_to_work_arrays(const mam4::Prognostics &progs,
                                         const int k, Real q[gas_pcnst()],
                                         Real qqcw[gas_pcnst()]) {
  DECLARE_PROG_TRANSFER_CONSTANTS

  // copy number/mass mixing ratios from progs to q and qqcw at level k,
  // converting them to VMR
  for (int i = 0; i < gas_pcnst(); ++i) {
    auto mode_index = mode_for_cnst[i];
    auto aero_id = aero_for_cnst[i];
    auto gas_id = gas_for_cnst[i];
    if (gas_id != NoGas) { // constituent is a gas
      int g = static_cast<int>(gas_id);
      q[i] = progs.q_gas[g](k);
      qqcw[i] = progs.q_gas[g](k);
    } else {
      int m = static_cast<int>(mode_index);
      if (aero_id != NoAero) { // constituent is an aerosol species
        int a = aerosol_index_for_mode(mode_index, aero_id);
        q[i] = progs.q_aero_i[m][a](k);
        qqcw[i] = progs.q_aero_c[m][a](k);
      } else { // constituent is a modal number mixing ratio
        int m = static_cast<int>(mode_index);
        q[i] = progs.n_mode_i[m](k);
        qqcw[i] = progs.n_mode_c[m](k);
      }
    }
  }
}

// Given work arrays with interstitial and cloudborne aerosol data, transfers
// them to the given Prognostics object at the kth vertical level. This is the
// "inverse operator" for transfer_prognostics_to_work_arrays, above.
KOKKOS_INLINE_FUNCTION
void transfer_work_arrays_to_prognostics(const Real q[gas_pcnst()],
                                         const Real qqcw[gas_pcnst()],
                                         mam4::Prognostics &progs,
                                         const int k) {
  DECLARE_PROG_TRANSFER_CONSTANTS

  // copy number/mass mixing ratios from progs to q and qqcw at level k,
  // converting them to VMR
  for (int i = 0; i < gas_pcnst(); ++i) {
    auto mode_index = mode_for_cnst[i];
    auto aero_id = aero_for_cnst[i];
    auto gas_id = gas_for_cnst[i];
    if (gas_id != NoGas) { // constituent is a gas
      int g = static_cast<int>(gas_id);
      progs.q_gas[g](k) = q[i];
    } else {
      int m = static_cast<int>(mode_index);
      if (aero_id != NoAero) { // constituent is an aerosol species
        int a = aerosol_index_for_mode(mode_index, aero_id);
        progs.q_aero_i[m][a](k) = q[i];
        progs.q_aero_c[m][a](k) = qqcw[i];
      } else { // constituent is a modal number mixing ratio
        int m = static_cast<int>(mode_index);
        progs.n_mode_i[m](k) = q[i];
        progs.n_mode_c[m](k) = qqcw[i];
      }
    }
  }
}

#undef DECLARE_PROG_TRANSFER_CONSTANTS

// given q and qqcs get arrays
// FIXME!!!: need aditional work.
KOKKOS_INLINE_FUNCTION
void transfer_tendencies_num_to_tendecines(const Real n_mode_i[],
                                           // const Real n_mode_c[],
                                           Real q[gas_pcnst()]
                                           // ,
                                           // Real qqcw[gas_pcnst()],
) {
  int s_idx = gasses_start_ind() +
              AeroConfig::num_gas_ids(); // gases start at index 9 (index 10 in
                                         // Fortran version)

  // Now start adding aerosols mmr into the state_q
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    s_idx += mam4::num_species_mode(m);
    q[s_idx] += n_mode_i[m];
    printf("q[%d] %e n_mode_i[%d] %e \n", s_idx, q[s_idx], m, n_mode_i[m]);
    s_idx++; // update index
    // printf(" %d ", s_idx);
  }

  // printf("\n ");
}

// return idx of num concentration in state_q
KOKKOS_INLINE_FUNCTION
void get_num_idx_in_state_q(int idxs[AeroConfig::num_modes()]) {
  // index of accum and aitken mode for num concetration in state_q
  int s_idx =
      gasses_start_ind() +
      AeroConfig::num_gas_ids(); // gases start at index 9 (index 10 in Fortran
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    s_idx += mam4::num_species_mode(m);
    idxs[m] = s_idx;
    s_idx++; // update index
  }
}

// Given an AerosolState with views for dry aerosol quantities, creates a
// interstitial aerosols 1D view (state_q) for the column with the given index.
// This object can be provided to mam4xx for the column.

// MUST FIXME: address James comments about making the code better.
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
extract_stateq_from_prognostics(const mam4::Prognostics &progs,
                                const Atmosphere &atm, VectorType &q,
                                const int klev) {

  int s_idx = ekat::invalid<int>();
  q[0] = atm.vapor_mixing_ratio(klev);               // qv
  q[1] = atm.liquid_mixing_ratio(klev);              // qc
  q[2] = atm.ice_mixing_ratio(klev);                 // qi
  q[3] = atm.cloud_liquid_number_mixing_ratio(klev); //  nc
  q[4] = atm.cloud_ice_number_mixing_ratio(klev);    // ni
  // FIXME: I do not have info for  :RAINQM, SNOWQM, NUMRAI, NUMSNO

  if (progs.q_gas[0].size()) { // if gases are defined in dry_aero aerosol state
    s_idx = gasses_start_ind(); // gases start at index 9 (index 10 in Fortran
                                // version)
    for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
      // get mmr at level "klev"
      q[s_idx] = progs.q_gas[g](klev);
      s_idx++; // update index
    }
  } else {
    s_idx = aero_start_ind(); // If no gasses; start with the first index of
                              // aerosols
  }

  // Now start adding aerosols mmr into the state_q
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    // First add the aerosol species mmr
    for (int a = 0; a < mam4::num_species_mode(m); ++a) {
      q[s_idx] = progs.q_aero_i[m][a](klev);
      s_idx++; // update index even if we lack some aerosol mmrs
    }
    q[s_idx] = progs.n_mode_i[m](klev);
    s_idx++; // update index
  }
} // extract_stateq_from_prognostics

template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
extract_ptend_from_tendencies(const Tendencies &tends, VectorType &ptend,
                              const int klev) {

  int s_idx = ekat::invalid<int>();
  // FIXME: tendencies for first five item (qv, qc, qi, nc, ni) should no be
  // modified by mam4xx is this correct ?
  if (tends.q_gas[0].size()) { // if gases are defined in dry_aero aerosol state
    s_idx = gasses_start_ind(); // gases start at index 9 (index 10 in Fortran
                                // version)
    for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
      // get mmr at level "klev"
      ptend[s_idx] = tends.q_gas[g](klev);
      s_idx++; // update index
    }
  } else {
    s_idx = aero_start_ind(); // If no gasses; start with the first index of
                              // aerosols
  }

  // Now start adding aerosols mmr into the state_q
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    // First add the aerosol species mmr
    for (int a = 0; a < mam4::num_species_mode(m); ++a) {
      ptend[s_idx] = tends.q_aero_i[m][a](klev);
      s_idx++; // update index even if we lack some aerosol mmrs
    }
    ptend[s_idx] = tends.n_mode_i[m](klev);
    s_idx++; // update index
  }
} // extract_ptend_from_tendencies
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
inject_stateq_to_prognostics(const VectorType &q, mam4::Prognostics &progs,
                             const int klev) {

  int s_idx = ekat::invalid<int>();

  if (progs.q_gas[0].size()) { // if gases are defined in dry_aero aerosol state
    s_idx = gasses_start_ind(); // gases start at index 9 (index 10 in Fortran
                                // version)
    for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
      // get mmr at level "klev"
      progs.q_gas[g](klev) = q[s_idx];
      s_idx++; // update index
    }
  } else {
    s_idx = aero_start_ind(); // If no gasses; start with the first index of
                              // aerosols
  }                           // end if

  // Now start adding aerosols mmr into the state_q
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    //   // First add the aerosol species mmr
    for (int a = 0; a < mam4::num_species_mode(m); ++a) {
      if (progs.q_aero_i[m][a].size()) {
        progs.q_aero_i[m][a](klev) = q[s_idx];
        s_idx++; // update index even if we lack some aerosol mmrs
      }          // end if
    }            // a
    // Now add aerosol number mmr
    progs.n_mode_i[m](klev) = q[s_idx];
    s_idx++; // update index
  }          // m
}
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void inject_ptend_to_tendencies(const VectorType &ptend,
                                                       const Tendencies &tends,
                                                       const int klev) {

  int s_idx = ekat::invalid<int>();

  if (tends.q_gas[0].size()) { // if gases are defined in dry_aero aerosol state
    s_idx = gasses_start_ind(); // gases start at index 9 (index 10 in Fortran
                                // version)
    for (int g = 0; g < AeroConfig::num_gas_ids(); ++g) {
      // get mmr at level "klev"
      tends.q_gas[g](klev) = ptend[s_idx];
      s_idx++; // update index
    }
  } else {
    s_idx = aero_start_ind(); // If no gasses; start with the first index of
                              // aerosols
  }                           // end if

  // Now start adding aerosols mmr into the state_q
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    //   // First add the aerosol species mmr
    for (int a = 0; a < mam4::num_species_mode(m); ++a) {
      if (tends.q_aero_i[m][a].size()) {
        tends.q_aero_i[m][a](klev) = ptend[s_idx];
        s_idx++; // update index even if we lack some aerosol mmrs
      }          // end if
    }            // a
    // Now add aerosol number mmr
    tends.n_mode_i[m](klev) = ptend[s_idx];
    s_idx++; // update index
  }          // m
}

// Given an AerosolState with views for dry aerosol quantities, creates a
// cloudborne aerosol mmr 1D view for the column with the given index.
// This object can be provided to mam4xx for the column.
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
extract_qqcw_from_prognostics(const mam4::Prognostics &progs, VectorType &qqcw,
                              const int klev) {

  // NOTE: qqcw view has the same dimension and indexing as state_q array.
  //  This array doesn't store gasses, so the indexing starts at aerosols

  // Initialize the start index of qqcw array
  int s_idx = aero_start_ind(); // If no gasses; start index with the first
                                // index of aerosols

  // Now start adding cloud borne aerosols mmr into the qqcw
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    // First add the aerosol species mmr
    for (int a = 0; a < mam4::num_species_mode(m); ++a) {
      if (progs.q_aero_c[m][a].size()) {
        qqcw[s_idx] = progs.q_aero_c[m][a](klev);
        s_idx++; // update index even if we lack some aerosol mmrs
      }
    } // a
    // Now add aerosol number mmr
    qqcw[s_idx] = progs.n_mode_c[m](klev);
    s_idx++; // update index
  }          // m
}

// Given an AerosolState with views for dry aerosol quantities, creates a
// cloudborne aerosol mmr 1D view for the column with the given index.
// This object can be provided to mam4xx for the column.
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void inject_qqcw_to_prognostics(const VectorType &qqcw,
                                                       mam4::Prognostics &progs,
                                                       const int klev) {

  // NOTE: qqcw view has the same dimension and indexing as state_q array.
  //  This array doesn't store gasses, so the indexing starts at aerosols

  // Initialize the start index of qqcw array
  int s_idx = aero_start_ind(); // If no gasses; start index with the first
                                // index of aerosols

  // Now start adding cloud borne aerosols mmr into the qqcw
  for (int m = 0; m < AeroConfig::num_modes(); ++m) {
    // First add the aerosol species mmr
    for (int a = 0; a < mam4::num_species_mode(m); ++a) {
      if (progs.q_aero_c[m][a].size()) {
        progs.q_aero_c[m][a](klev) = qqcw[s_idx];
        s_idx++; // update index even if we lack some aerosol mmrs
      }
    }
    // Now add aerosol number mmr
    EKAT_KERNEL_ASSERT_MSG(progs.n_mode_c[m].size(),
                           "cld_aero_nmr not defined for dry aerosol state!");
    progs.n_mode_c[m](klev) = qqcw[s_idx];
    s_idx++; // update index
  }
}

} // end namespace mam4::utils

#endif
