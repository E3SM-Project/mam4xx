#ifndef MAM4XX_GAS_CHEM_HPP
#define MAM4XX_GAS_CHEM_HPP

#include "mam4_math.hpp"
#include <mam4xx/mam4_config.hpp>

#include <ekat_kokkos_types.hpp>

// pp_linoz_mam4_resus_mom_soag
// Generated code.
// Authors: Oscar Diaz-Ibarra (odiazib@sandia.gov)
//          Mike Schmidt (mjschm@sandia.gov)
//
// Reaction stoichiometry and coefficients were ported from the legacy EAM
// pp_linoz_mam4_resus_mom_soag mechanism. Its generated chem_mech.doc records
// the formulas below but not literature citations. The user-defined R1, R3,
// and R5 formulas come from legacy mo_usrrxt.F90; see gas_chem.hpp.
// Reaction notation and units:
// T is temperature [K], M is total air number density [molecules cm^-3],
// and [X] is the prescribed number density of reactant X [molecules cm^-3].
// q_X is the molar mixing ratio of gas X. The effective bimolecular rate
// coefficient k [cm^3 molecule^-1 s^-1], multiplied by a prescribed oxidant
// density, gives the fixed pseudo-first-order coefficient lambda [s^-1].
// Aerosol-mass and modal-aerosol-number species share the chemistry work array;
// their sources and losses use the corresponding work-array entry unit s^-1.
namespace mam4 {
namespace gas_chemistry {

constexpr int nabscol = 2;    // number of absorbing densities
constexpr int rxntot = 7;     // number of total reactions
constexpr int gas_pcnst = 31; // gas, aerosol-mass, and modal-number work entries
constexpr int nzcnt = 32;     // number of non-zero matrix entries
constexpr int clscnt4 = 30;   // work entries 1-30; O3 (entry 0) is excluded
constexpr int extcnt = 9;     // number of species with external forcing
constexpr int nfs = 8;        // number of fixed species
constexpr int o3_idx = 0;     // index of O3
constexpr int indexm = 0;     // total air number density [molecules cm^-3]
constexpr auto permute_4 = Kokkos::to_array<int>(
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
constexpr auto clsmap_4 = Kokkos::to_array<int>(
    {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});
constexpr auto adv_mass = Kokkos::to_array<Real>(
    {47.998200,     34.013600,  98.078400,     64.064800, 62.132400,
     12.011000,     115.107340, 12.011000,     12.011000, 12.011000,
     135.064039,    58.442468,  250092.672000, 1.007400,  115.107340,
     12.011000,     58.442468,  250092.672000, 1.007400,  135.064039,
     58.442468,     115.107340, 12.011000,     12.011000, 12.011000,
     250092.672000, 1.007400,   12.011000,     12.011000, 250092.672000,
     1.007400});

KOKKOS_INLINE_FUNCTION
void setrxt(Real rates[rxntot], const Real temp) {
  // temp is in K. The coefficients below are in cm^3 molecule^-1 s^-1;
  // the constants divided by temp inside exp have temperature units (K).
  // R2: H2O2 + OH -> H2O + HO2
  // Bimolecular coefficient: k2(T) = 2.9e-12 * exp(-160/T).
  rates[2] = 2.9000000000e-12 * mam4::exp(-160.000000 / temp);

  // R4: DMS + OH -> SO2
  // Bimolecular coefficient: k4(T) = 9.6e-12 * exp(-234/T).
  rates[4] = 9.6000000000e-12 * mam4::exp(-234.000000 / temp);

  // R6: DMS + NO3 -> SO2 + HNO3
  // Bimolecular coefficient: k6(T) = 1.9e-13 * exp(520/T).
  rates[6] = 1.9000000000e-13 * mam4::exp(520.000000 / temp);
} // setrxt

KOKKOS_INLINE_FUNCTION
void set_rates(Real rxt_rates[rxntot], Real sol[gas_pcnst]) {
  // On entry, slots 0 and 2-6 are first-order coefficients [s^-1]. Multiply
  // by gas mixing ratios to form completed mixing-ratio tendencies per second.
  // Slot 1 is already the state-independent H2O2 source and is left unchanged.
  // J1: H2O2 + hv -> products not retained by this mechanism.
  // Completed loss rate: J(H2O2) * q_H2O2.
  rxt_rates[0] *= sol[1];

  // R2: H2O2 + OH -> H2O + HO2.
  // Completed loss rate: k2(T) * [OH] * q_H2O2.
  rxt_rates[2] *= sol[1];

  // R3: SO2 + OH -> H2SO4.
  // Completed transfer rate: k3(T,M) * [OH] * q_SO2.
  rxt_rates[3] *= sol[3];

  // R4: DMS + OH -> SO2.
  // Completed transfer rate: k4(T) * [OH] * q_DMS.
  rxt_rates[4] *= sol[4];

  // R5: DMS + OH -> 0.5 SO2 + 0.5 HO2.
  // Completed DMS loss rate: k5(T,M) * [OH] * q_DMS; SO2 receives half.
  rxt_rates[5] *= sol[4];

  // R6: DMS + NO3 -> SO2 + HNO3.
  // Completed transfer rate: k6(T) * [NO3] * q_DMS.
  rxt_rates[6] *= sol[4];
} // set_rates

KOKKOS_INLINE_FUNCTION
void adjrxt(Real rate[rxntot], const Real inv[nfs], const Real m) {
  // Convert the bimolecular coefficients into frozen pseudo-first-order
  // coefficients [s^-1] using prescribed densities [molecules cm^-3].
  // inv[4]=OH, inv[5]=NO3, inv[6]=HO2; m is air number density [molecules cm^-3].
  // R2: lambda2 = k2(T) * [OH].
  rate[2] *= inv[4];
  // R3: lambda3 = k3(T,M) * [OH].
  rate[3] *= inv[4];
  // R4: lambda4 = k4(T) * [OH].
  rate[4] *= inv[4];
  // R5: lambda5 = k5(T,M) * [OH].
  rate[5] *= inv[4];
  // R6: lambda6 = k6(T) * [NO3].
  rate[6] *= inv[5];

  // R1: HO2 + HO2 -> H2O2.
  // Completed mixing-ratio source: P1 = k1(T,M,H2O) * [HO2]^2 / M.
  // k1*[HO2]^2 is molecules cm^-3 s^-1; division by M gives a
  // mixing-ratio tendency [s^-1], not another first-order loss coefficient.
  rate[1] *= inv[6] * inv[6] / m;
} // adjrxt

// Calculate production and loss at state y for the 30 implicit work entries.
// Output index k corresponds to y[k+1]; O3 is excluded. The caller adds the
// state-independent sources from indprd to prod afterward.
// For inputs prepared by gas_phase_chemistry, rxt[0] and rxt[2] through rxt[6]
// are loss or transfer coefficients in s^-1; het_rates are heterogeneous
// first-order removal coefficients in s^-1, not concentration tendencies.
// Multiplication by y gives rates in the corresponding state-entry units per
// second (molar mixing ratio per second for gases). rxt[1] is an H2O2 source,
// not a first-order coefficient, and is included by indprd rather than here.
template <typename VectorType>
KOKKOS_INLINE_FUNCTION void
imp_prod_loss(Real prod[clscnt4], Real loss[clscnt4], const VectorType &y,
              const Real rxt[rxntot], const Real het_rates[gas_pcnst]) {
  const Real zero = 0;
  loss[0] = (het_rates[1] + rxt[0] + rxt[2]) * y[1];
  prod[0] = zero;
  loss[1] = het_rates[2] * y[2];
  prod[1] = rxt[3] * y[3];
  loss[2] = (het_rates[3] + rxt[3]) * y[3];
  prod[2] = (rxt[4] + 0.500000 * rxt[5] + rxt[6]) * y[4];
  loss[3] = (het_rates[4] + rxt[4] + rxt[5] + rxt[6]) * y[4];
  prod[3] = zero;
  for (int i = 4; i < clscnt4; ++i) {
    loss[i] = het_rates[i + 1] * y[i + 1];
    prod[i] = zero;
  }
} // imp_prod_loss

KOKKOS_INLINE_FUNCTION
void indprd(const int class_id, Real prod[clscnt4], const Real rxt[rxntot],
            const Real extfrc[extcnt]) {
  // Sources that do not depend on the updated species values. The caller
  // gas_phase_chemistry divides external forcing by air number density before
  // passing it here. prod and extfrc therefore use work-entry units per second.
  // For gases, this is mixing-ratio s^-1, not molecules cm^-3 s^-1.
  const Real zero = 0;
  // Class 1 contains O3, whose source is zero in this chemistry operator.
  // Class 4 contains the 30 implicit entries; imp_sol requests this class.
  if (class_id == 1) {
    prod[0] = zero;
  } else if (class_id == 4) {
    prod[0] = +rxt[1];
    prod[1] = zero;
    prod[2] = +extfrc[0];
    prod[3] = zero;
    prod[4] = +extfrc[8];
    prod[5] = +extfrc[1];
    prod[6] = zero;
    prod[7] = zero;
    prod[8] = zero;
    prod[9] = zero;
    prod[10] = zero;
    prod[11] = zero;
    prod[12] = +extfrc[5];
    prod[13] = +extfrc[2];
    prod[14] = zero;
    prod[15] = zero;
    prod[16] = zero;
    prod[17] = +extfrc[6];
    prod[18] = zero;
    prod[19] = zero;
    prod[20] = zero;
    prod[21] = zero;
    prod[22] = zero;
    prod[23] = zero;
    prod[24] = zero;
    prod[25] = zero;
    prod[26] = +extfrc[3];
    prod[27] = +extfrc[4];
    prod[28] = zero;
    prod[29] = +extfrc[7];
  } // indprd
}

// Build the 32 stored coefficients of dq/dt = A*q + b for the 30 implicit
// unknowns. The six leading entries represent four diagonal losses plus the
// SO2->H2SO4 and DMS->SO2 couplings; the rest are independent diagonal losses.
// A has units s^-1, so A*q and b both have the state-entry unit s^-1.
// Diagonal entries are negative total loss coefficients; off-diagonals are
// positive transfer coefficients. Multiplying an off-diagonal by the upstream
// species mixing ratio gives the receiving species' production tendency.
// rxt[0], rxt[2] through rxt[6], and het_rates are coefficients in s^-1 for
// inputs prepared by gas_phase_chemistry. The H2O2 source rxt[1] belongs to b
// and is supplied by indprd, so it does not appear in A.
KOKKOS_INLINE_FUNCTION
void linmat(Real mat[nzcnt], const Real rxt[rxntot],
            const Real het_rates[gas_pcnst]) {
  mat[0] = -(+rxt[0] + rxt[2] + het_rates[1]);
  mat[1] = -(+het_rates[2]);
  mat[2] = +rxt[3];
  mat[3] = -(+rxt[3] + het_rates[3]);
  mat[4] = +rxt[4] + 0.500000 * rxt[5] + rxt[6];
  mat[5] = -(+rxt[4] + rxt[5] + rxt[6] + het_rates[4]);
  for (int k = 4; k < clscnt4; ++k) {
    mat[k + 2] = -het_rates[k + 1];
  }
} // linmat

} // namespace gas_chemistry
} // namespace mam4
#endif
