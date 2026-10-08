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
// In the explanatory comments, T is temperature, M is total air number
// density, [X] is a prescribed reactant number density, q_X is the chemistry
// work-array value, k is a reaction coefficient, and lambda is a frozen
// pseudo-first-order coefficient.
namespace mam4 {
namespace gas_chemistry {

constexpr int nabscol = 2;    // number of absorbing densities
constexpr int rxntot = 7;     // number of total reactions
constexpr int gas_pcnst = 31; // number of gas phase species
constexpr int nzcnt = 32;     // number of non-zero matrix entries
constexpr int clscnt4 = 30;   // number of species in implicit class
constexpr int extcnt = 9;     // number of species with external forcing
constexpr int nfs = 8;        // number of fixed species
constexpr int o3_idx = 0;     // index of O3
constexpr int indexm = 0;     // index of total atm density in invariant array
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
  // coefficients using prescribed oxidant number densities.
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
  rate[1] *= inv[6] * inv[6] / m;
} // adjrxt

// TODO: unless rxt[0:6] and/or het_rates[1:4] have different units than the
// rest of the arrays the below additions seem fishy
// Units:
// rxt := reaction rates in 1D array [1/cm^3/s]
// het_rates := washout rates [1/s]
// TODO: the lines of concern *kind of* bear resemblance to the similarly
// concerning lines in linmat(), though it's difficult to tell if that results
// in consistent units
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
  // extfrc := external in-situ forcing [1/cm^3/s]
  // thus, prod must have units [1/cm^3/s]
  const Real zero = 0;
  // this is hard-coded to 4 outside of this function
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

// TODO: as in imp_prod_loss() unless rxt[0:6] and/or het_rates[1:4] have
// different units than the rest of the arrays the below additions seem suspect
// Units:
// rxt := reaction rates in 1D array [1/cm^3/s]
// het_rates := washout rates [1/s]
// TODO: the lines of concern *kind of* bear resemblance to the similarly
// concerning lines in imp_prod_loss(), though it's difficult to tell if that
// results in consistent units
KOKKOS_INLINE_FUNCTION
void linmat(Real mat[nzcnt], const Real rxt[rxntot],
            const Real het_rates[gas_pcnst]) {
  mat[0] = -(+rxt[0] + rxt[2] + het_rates[1]);
  mat[1] = -(+het_rates[2]);
  mat[2] = +rxt[3];
  mat[3] = -(+rxt[3] + het_rates[3]);
  mat[4] = +rxt[4] + 0.500000 * rxt[5] + rxt[6];
  mat[5] = -(+rxt[4] + rxt[5] + rxt[6] + het_rates[4]);
  mat[6] = -(+het_rates[5]);
  mat[7] = -(+het_rates[6]);
  mat[8] = -(+het_rates[7]);
  mat[9] = -(+het_rates[8]);
  mat[10] = -(+het_rates[9]);
  mat[11] = -(+het_rates[10]);
  mat[12] = -(+het_rates[11]);
  mat[13] = -(+het_rates[12]);
  mat[14] = -(+het_rates[13]);
  mat[15] = -(+het_rates[14]);
  mat[16] = -(+het_rates[15]);
  mat[17] = -(+het_rates[16]);
  mat[18] = -(+het_rates[17]);
  mat[19] = -(+het_rates[18]);
  mat[20] = -(+het_rates[19]);
  mat[21] = -(+het_rates[20]);
  mat[22] = -(+het_rates[21]);
  mat[23] = -(+het_rates[22]);
  mat[24] = -(+het_rates[23]);
  mat[25] = -(+het_rates[24]);
  mat[26] = -(+het_rates[25]);
  mat[27] = -(+het_rates[26]);
  mat[28] = -(+het_rates[27]);
  mat[29] = -(+het_rates[28]);
  mat[30] = -(+het_rates[29]);
  mat[31] = -(+het_rates[30]);
} // linmat

} // namespace gas_chemistry
} // namespace mam4
#endif
