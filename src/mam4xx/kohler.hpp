// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MAM4XX_KOHLER_HPP
#define MAM4XX_KOHLER_HPP

#include "floating_point.hpp"
#include "mam4_constants.hpp"
#include "mam4_math.hpp"

namespace mam4 {

/// Surface tension of liquid water in air as a function of temperature
///   @param [in] T temperature [K]
///   @return sigma [N/m]
///
///   Called without an argument, the default input reproduces the value used by
///   MAM4's approximation of constant surface tension, neglecting temperature
///   dependence.
///
///   This formula is valid from T = 248.16 K (-25 C, supercooled liquid water)
///   to the critical temperature Tc = 646.096 K (steam).
///
///   IAPWS Release on Surface Tension of Ordinary Water Substance
///   IAPWS R1-76(2014)
///   http://www.iapws.org/relguide/Surf-H2O.html
///
///   Referenced from: NIST WebBook Saturation Properties for Water
///   https://webbook.nist.gov/cgi/fluid.cgi?TLow=273.16&THigh=343&TInc=1&Applet=on&Digits=5&ID=C7732185&Action=Load&Type=SatP&TUnit=K&PUnit=MPa&DUnit=mol%2Fl&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm&RefState=DEF
///
///   accessed on September 20, 2022
KOKKOS_INLINE_FUNCTION double
surface_tension_water_air(double T = Constants::triple_pt_h2o) {
  constexpr double Tc = Constants::tc_water;
  constexpr double B = 0.2358;
  constexpr double b = -0.625;
  constexpr double mu = 1.256;
  const auto tau = 1 - T / Tc;
  EKAT_KERNEL_ASSERT(
      FloatingPoint<double>::in_bounds(T, Constants::triple_pt_h2o - 25, Tc,
                                       Kokkos::Experimental::epsilon<float>::value));
  return B * mam4::pow(tau, mu) * (1 + b * tau);
}

/// Kelvin coefficient
///   Equation (A1) from Ghan et al., 2011, Droplet nucleation: Physically-based
///    parameterizations and comparative evaluation, J. Adv. Earth Sys. Mod. 3
///    M10001.
///
///   To reproduce MAM4 approximations, call this function without an argument;
///   the default value is set to the constant value used by MAM4.
///
///   ** Note: This subroutine uses SI units **
///
///   MAM4 uses micrometers and milinewtons.
///
///   @param [in] T temperature [K]
///   @return Kelvin droplet coefficient [m]
KOKKOS_INLINE_FUNCTION double
kelvin_coefficient(double T = Constants::triple_pt_h2o) {
  const double density_h2o = Constants::density_h2o;
  const double r_gas_h2o_vapor = Constants::r_gas_h2o_vapor;
  return 2 * surface_tension_water_air(T) / (r_gas_h2o_vapor * T * density_h2o);
}

/// Struct that represents the Kohler polynomial.
///
///   @f$ K(r_w) = \log(s) r_w^4 - A r_w^3 + (B - \log(s))r_d^3 r_w + A r_d^3
///   @f$
///
///   where r_w is the wet radius, s is relative humidity, A is the Kelvin
///   effect coefficient, B is hygroscopicity, and r_d is the dry radius.
///
///   The Kohler polynomial is a quartic polynomial whose variable is particle
///   wet radius. Equilibrium solutions are roots of this polynomial.
///   Algebraically, there are 2 complex roots and 2 real roots.  Of the real
///   roots, one is positive, the other is negative. Physically, only the real,
///   positive root makes sense.
///
///   Each instance corresponds to a separate set of coefficients, which are
///   functions of the inputs.
///
///   This struct conforms to the interface prescribed in math.hpp for
///   scalar functions that are to be used with numerical rootfinding
///   algorithms.
///
///   @warning This polynomial is severely ill-conditioned, to the point that it
///   is sensitive to order-of-operations changes caused by compiler
///   optimization flags.  We therefore require double precision.
///
///   Properties of the Kohler Polynomial that are useful to finding its roots:
///
///   1. K(0) = kelvin_droplet_effect_coeff * cube(r_dry) > 0
///   2. K(r_dry) = r_dry**4 * hygroscopicity > 0
///
///   Properties of the Kohler Polynomial that are useful to finding its roots
///   given inputs that are within the bounds defined below:
///
///   1. K(25*r_dry) < 0
struct KohlerPolynomial {
  using value_type = double; // double precision required

  /// Minimum value of relative humidity
  static constexpr double rel_humidity_min = 0.05;
  /// Above this relative humidity is considered saturated air, and cloud
  /// aerosol processes would apply
  static constexpr double rel_humidity_max = 0.98;
  /// Minimum hygroscopicity for E3SM v1 aerosol species
  static constexpr double hygro_min = 1e-6;
  /// Maximum hygroscopicity for E3SM v1 aerosol species
  static constexpr double hygro_max = 1.3;
  /// Minimum particle size for E3SM v1
  static constexpr double dry_radius_min_microns = 1e-3;
  /// Maximum particle size for Kohler theory
  static constexpr double dry_radius_max_microns = 30;

  /// Coefficient in the Kohler polynomial
  double log_rel_humidity;
  /// Coefficient in the Kohler polynomial
  double hygroscopicity;
  /// Safe return value
  double dry_radius;
  /// Coefficient in the Kohler polynomial
  double dry_radius_cubed;
  /// Coefficient in the Kohler polynomial
  double kelvin_a;

  /** Constructor. Creates 1 instance of a KohlerPolynomial.

    @param [in] rel_h relative humidity
    @param [in] hygro hygroscopicity
    @param [in] dry_rad_microns particle dry radius [ 1e-6 m ]
    @param [in] temperature [K]
  */
  KOKKOS_INLINE_FUNCTION
  KohlerPolynomial(Real rel_h, Real hygro, Real dry_rad_microns,
                   Real temperature = Constants::triple_pt_h2o)
      : log_rel_humidity(mam4::log(rel_h)), hygroscopicity(hygro),
        dry_radius(dry_rad_microns),
        dry_radius_cubed(mam4::cube(dry_rad_microns)),
        kelvin_a(kelvin_coefficient(temperature)) {

    kelvin_a *= 1e6; /* convert from N to mN and m to micron */
    EKAT_KERNEL_ASSERT(valid_inputs(rel_h, hygro, dry_rad_microns));
  }

  ///   Evaluates the Kohler polynomial.
  ///
  ///     @f$ K(r_w) = \log(s) r_w^4 - A r_w^3 + (B - \log(s))r_d^3 r_w + A
  ///     r_d^3 @f$
  ///
  ///     where r_w is the wet radius, s is relative humidity, A is the Kelvin
  ///     effect coefficient, B is hygroscopicity, and r_d is the dry radius.
  ///
  ///     @param [in] Polynomial input value, wet_radius wet radius in microns [
  ///     1e-6 m]
  ///     @return Polynomial value, wet_radius in microns [ 1e-6 m]
  template <typename U>
  KOKKOS_INLINE_FUNCTION double operator()(const U &wet_radius) const {
    const double rwet = Real(wet_radius);
    const double result =
        (log_rel_humidity * rwet - kelvin_a) * cube(rwet) +
        ((hygroscopicity - log_rel_humidity) * rwet + kelvin_a) *
            dry_radius_cubed;
    return result;
  }

  ///   Evaluates the derivative of the Kohler polynomial with respect to
  ///     wet radius
  ///
  ///     @f$ K'(r_w) = \frac{\partial K}(\partial r_w)(r_w) @f$
  ///
  ///     @param [in] Polynomial input value, wet radius in microns [ 1e-6 m]
  ///     @return Polynomial slope at input value
  template <typename U>
  KOKKOS_INLINE_FUNCTION double derivative(const U &wet_radius) const {
    const double rwet = double(wet_radius);
    const double wet_radius_squared = square(rwet);
    const double result =
        (4 * log_rel_humidity * rwet - 3 * kelvin_a) * wet_radius_squared +
        (hygroscopicity - log_rel_humidity) * dry_radius_cubed;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  bool valid_inputs(double relh, double hyg, double dry_rad) const {
    return (FloatingPoint<double>::in_bounds(relh, rel_humidity_min,
                                             rel_humidity_max) and
            FloatingPoint<double>::in_bounds(hyg, hygro_min, hygro_max) and
            FloatingPoint<double>::in_bounds(dry_rad, dry_radius_min_microns,
                                             dry_radius_max_microns));
  }

  KOKKOS_INLINE_FUNCTION
  bool valid_inputs() const {
    return valid_inputs(mam4::exp(this->log_rel_humidity), this->hygroscopicity,
                        this->dry_radius);
  }
};

/// Solver for the Kohler polynomial; templated so that it can
/// use any root finding algorithm from the math namespace.
///
/// This solver replaces subroutine modal_aero_kohler from
/// modal_aero_wateruptake.F90.
template <typename SolverType> struct KohlerSolver {
  using value_type = double;
  double relative_humidity;
  double hygroscopicity;
  double dry_radius_microns;
  Real conv_tol;
  int n_iter;

  KOKKOS_INLINE_FUNCTION
  KohlerSolver(Real rel_h, Real hyg, Real rdry, Real tol)
      : relative_humidity(rel_h), hygroscopicity(hyg), dry_radius_microns(rdry),
        conv_tol(tol), n_iter(0) {}

  KOKKOS_INLINE_FUNCTION
  double solve() {
    double wet_radius_left = 0.9 * dry_radius_microns;
    double wet_radius_right = 50 * dry_radius_microns;
    double wet_radius_init = 25 * dry_radius_microns;
    const Real triple_pt_h2o = Constants::triple_pt_h2o;
    const double default_T = triple_pt_h2o;
    const auto kpoly = KohlerPolynomial(relative_humidity, hygroscopicity,
                                        dry_radius_microns, default_T);
    auto solver = SolverType(wet_radius_init, wet_radius_left, wet_radius_right,
                             conv_tol, kpoly);
    const double result = solver.solve();
    n_iter = solver.counter;
    return result;
  }
};

} // namespace mam4
#endif
