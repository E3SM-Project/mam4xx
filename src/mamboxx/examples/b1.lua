-- B1: activation from a known aerosol spectrum.
-- DISCUSSION EXAMPLE for Jeff; proposed schema, not an executable input for an
-- existing driver. Retains Jeff's time/control/atmosphere/aerosols/gases layout.
-- New controls, size inputs, sweeps, output names, and checks need driver support.
--
-- Sources:
-- Jeff's pseudo-example, 2026-10-29:
-- https://grid-cels-anl.enterprise.slack.com/archives/D07J3PB0L5Q/p1793298098207339
-- B1 definition, Confluence version 3, 2026-10-27:
-- https://e3sm.atlassian.net/wiki/spaces/EAMXX/pages/6814826520
-- Kappa-Kohler notation/mixing rule: Petters and Kreidenweis (2007), Sec. 2:
-- https://acp.copernicus.org/articles/7/1961/2007/
-- Constants/properties checked in local MAM4xx df490c4775e0b104514cd421429f5fa22b247460:
-- src/mam4xx/{ndrop,aero_config,aero_modes,mam4_constants}.hpp
--
-- One B1 example: pure sulfate, D_g=100 nm, sigma_g=1.8, N=100 cm^-3.
-- B1's figure parameters are illustrative: this example uses kappa=0.507 and
-- T=273 K rather than the figure's kappa=0.3 and T=283 K. Pure sulfate avoids
-- ambiguity in mixture hygroscopicity; 273 K also matches ndrop_init's fixed
-- curvature temperature. This specifies a test, not a change to model physics.
--
-- Deferred follow-up (2026-10-29; OPEN CONCERN, not an established defect):
-- Audit activation input preparation in ndrop::{get_aer_num,loadaer} and its
-- downstream callers for consistency and possible conservation effects.
-- get_aer_num bounds a local number concentration using dry volume and modal
-- diameter limits; this helper does not itself overwrite prognostic mass.
-- Record raw/used number, volume, inferred diameter, and hygroscopicity; trace
-- how the resulting fractions feed droplet sources and aerosol phase transfers.
-- Check interstitial + cloud-borne dry-species mass budgets and number transfers
-- with size bounds active/inactive, including empty modes and positivity guards.
-- Distinguish local diagnostic changes from changes to prognostic state. No mass
-- conservation failure has been demonstrated by the present source inspection.
-- For B1's analytical comparison, require the original population to reach the
-- activation calculation unchanged within roundoff; do not disable safeguards
-- or rebuild the reference from a silently adjusted population to obtain a pass.

-------------------------
-- Verification checks --
-------------------------

-- These check functions are called by the box model at the end of a simulation.
-- Each of these check functions accepts an aerosol state and returns true if
-- the state passes the check, false if not. We can minimize unnecessary comments
-- by selecting appropriate names.

function physical_constants_are_correct(state)
  -- constants are always available in the constants table
  return constants.curvature_temperature == 273.0 and
         constants.surface_tension == 0.076 and
         constants.water_molar_mass == 0.018016 and
         constants.water_density == 1000.0 and
         constants.gas_constant == 8.314563841815325
end

function number_and_mass_fractions_match_independent_references(state)
  -- for each relevant aerosol mode...
  return state.prognostics.modes.aerosol_mode.number_fraction == mode_number_fraction_ref and
         state.prognostics.modes.aerosol_mode.mass_fractions.<species> == mode_species_mass_fraction_ref
end

function no_clipping_occurred(state)
  return state.diagnostics.num_clips == 0
end

function mass_and_number_fractions_increase_monatonically_with_supersaturation(state)
  -- this can be done by constructing sensitivities from the various runs
end

function number_fraction_is_one_half_at_critical_supersaturation_of_median_diameter(state)
  -- etc
end

--------------------------
-- Box model invocation --
--------------------------

box{
  -- experiment design (single-parameter sweep)
  experiment = {
    name = 'B1_single_mode_sulfate',
    sweep = {
      -- RH relative to liquid water, expressed as a fraction, NOT percent.
      -- RH=1.002 means 100.2% RH and fractional supersaturation s=0.002.
      relative_humidity = {1.0, 1.0002, 1.0005, 1.001, 1.002, 1.0020259974297443, 1.005, 1.01},
      reset_complete_state = true, -- FIXME: what is this?
    },
  },

  -- timestepping
  time = {
    dt    = 1,  -- [s]; bookkeeping only for this diagnostic test
    nstep = 1,  -- single independent evaluation per sweep value
  },

  -- enabled processes
  control = {
    droplet_activation = true,
    gas_chemistry = false,
    gas_aerosol_exchange = false,
    rename = false,
    nucleation = false,
    coagulation = false,
    aqueous_chemistry = false,
    aerosol_water_uptake = false,
    emissions = false,
    dry_deposition = false,
    wet_deposition = false,
    vertical_mixing = false,
    precipitation = false,
    sedimentation = false,
    ice_processes = false,
  },

  -- atmospheric conditions (all parameters held fixed)
  -- Prescribe RH relative to liquid water and hold it fixed at each sweep point.
  -- Derive fractional supersaturation s=RH-1; no prognostic vapor depletion.
  atmosphere = {
    temperature = 273,                 -- [K]
    pressure = 100000,                 -- [Pa]
    initial_cloud_droplet_number = 0,  -- [m^-3]
  },

  aerosols = {
    -- For dry mass fractions w_j and material densities rho_j:
    -- rho_mix = 1 / sum_j(w_j/rho_j).
    -- V_total = N*(pi/6)*D_g^3*exp(4.5*ln(sigma_g)^2) [m^3 dry aerosol/m^3 air].
    -- M_total = rho_mix*V_total [kg/m^3]; M_j = w_j*M_total.
    -- Thus size, width, number, and species masses describe the SAME population.
    -- For this pure-sulfate case: V_total=2.4785895091943015e-13 m^3/m^3,
    -- M_total=M_so4=4.3871034312739137e-10 kg/m^3.
    -- If the backend needs dry-air mixing ratios, divide concentrations by the
    -- same diagnosed dry-air density and record that density and conversion.

    -- NOTE: these things are all fixed either at compile time or within mam4xx
    --concentration_basis = 'per_m3_air',
    --composition_basis   = 'dry_mass_fraction',
    --distribution = 'untruncated_lognormal',
    --initialization = 'derive_species_masses_from_number_size_composition',

    -- missing_species = zero  NOTE: this and the next line should be a policy for the driver (or not)
    -- empty_modes =  exact_zero       # no number/mass floors or division by zero

    initial_cloud_borne_number = 0,       -- [m^-3], all modes
    initial_cloud_borne_species_mass = 0, -- [kg/m^3], all mode/species pairs
    required_species_properties = {  -- assertions on backend values, NOT overrides
      so4 = {
        density = 1770   -- [kg/m^3]
        kappa   = 0.507  -- [-]; uniform within this mode
      },
    },
    modes = {
      accumulation = {
        number_concentration         = 1e8,  -- [#/m^-3]
        dry_number_median_diameter   = 1e-7, -- [m]
        geometric_standard_deviation = 1.8,  -- [-]
        mass_fractions = {
          so4 = 1,
        },
      },
      aitken = {
        number_concentration = 0
      },
      coarse = {
        number_concentration = 0,
      },
      primary_carbon = {
        number_concentration = 0,
      },
    },
  },

  gases = {
    so2  = 0,
    so4  = 0,
    soag = 0,
  },

  -- Evaluate the production activation-fraction mapping at s=RH-1, bypassing
  -- maxsat and velocity gates. Report eligible N and dry mass over the entire
  -- aerosol population, without cloud-fraction weighting or reservoir updates.
  -- This is equilibrium eligibility, not finite-time growth or a flux test.
  droplet_activation = {
    -- NOTE: These are likely fixed by mam4xx in the first place
    --supersaturation_source = 'derived_from_relative_humidity',
    --updraft_supersaturation_closure: 'disabled',
    --evaluation: 'diagnostic_only',

    critical_relation = 'dilute_kappa_kohler',
    -- Diameter convention: A=4*sigma*M_w/(R*T*rho_w)=2.4128472972784091e-9 m.
    -- A kernel using the radius convention must use a=A/2 with matched constants.
  },

  output = {
    filename = 'B1_single_mode_sulfate.csv',
    --NOTE: all fields should be written out for box and column models

    --NOTE: these should all be done always to the extent possible
    --significant_digits = 17,
    --write_initial_state = true,
    --write_requested_and_used_inputs = true,
  },

  -- Compute in a separate checker, not by calling the tested activation kernel.
  -- Derive s=RH-1 from the actual stored RH for each sweep point, including
  -- floating-point subtraction roundoff; do not assume exact decimal s values.
  -- For s>0: D_c=(4*A^3/(27*kappa*s^2))^(1/3), L=ln(sigma_g),
  -- f_N=0.5*erfc(ln(D_c/D_g)/(sqrt(2)*L)),
  -- f_M=0.5*erfc((ln(D_c/D_g)-3*L^2)/(sqrt(2)*L)).
  -- At s=0 use the exact limit f_N=f_M=0 and D_c=+infinity;
  -- avoid evaluating log(0). All other diagnostics must remain finite.
  verification = {
    reference = 'independent_lognormal_tail_integrals',
    -- NOTE: the ordering of these expected quantities is determined by the ordering of the parameters
    -- NOTE: in the sweep(s) specified in the `experiment` section. This is not practical for multiple-
    -- NOTE: parameter sweeps without further specification.
    expected_number_fractions = {
      0.0, 0.004316742447384711, 0.056258485016730392, 0.21161951810137772,
      0.49415645356905025, 0.49999999999999667, 0.84722574433828679, 0.96491205591080664,
    },

    expected_dry_mass_fractions = {
      0.0, 0.19410272871029796, 0.57000193549259004, 0.83211216604095939,
      0.95982958159524401, 0.96108010372429453, 0.99734800522073563, 0.99982430532297384,
    },

    tolerance = {
      roundoff_multiplier = 512,  -- multiple of floating point epsilon to use for tolerance
    },

    -- Checks are functions that evaluate expressions of available quantities, returning true on
    -- success and false on failure. The final simulation state is passed to the function in a
    -- `state` table with prognostics and diagnostics entries. Constants are available in a
    -- `constants` table.
    -- NOTE: mass fractions always fall within the interval [0, 1].
    -- NOTE: diagnostic_evaluation_leaves_both_aerosol_reservoirs_unchanged -- verified manifestly
    -- NOTE: all_unrequested_processes_are_disabled -- verified manifestly
    -- NOTE: initialized_mass_and_inferred_size_match_the_lognormal_moment -- verified manifestly
    -- NOTE: empty_modes_have_zero_activated_amounts_without_undefined_fractions -- verified manifestly
    checks = {
      physical_constants_are_correct,
      number_and_mass_fractions_match_independent_references,
      no_clipping_occurred,
      mass_and_number_fractions_increase_monatonically_with_supersaturation,
      number_fraction_is_one_half_at_critical_supersaturation_of_median_diameter,
    },
  },
}
