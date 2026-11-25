# C4: Preservation of uniform initial condition

import numpy as np

#---------------------
# Verification checks
#---------------------

# These check functions are called by the box model at the end of a simulation.
# Each of these check functions accepts an aerosol state and returns true if
# the state passes the check, false if not. We can minimize unnecessary comments
# by selecting appropriate names.

def mixing_ratios_remain_invariant(state):
    modes = state.prognostics.modes
    return np.all(modes.accumulation.so4 == modes.accumulation.so40) and
           np.all(modes.accumulation.pom == modes.accumulation.pom0) and
           np.all(modes.accumulation.soa == modes.accumulation.soa0) and
           np.all(modes.accumulation.bc  == modes.accumulation.bc0 ) and
           np.all(modes.accumulation.dst == modes.accumulation.dst0) and
           np.all(modes.accumulation.ncl == modes.accumulation.ncl0) and
           np.all(modes.accumulation.mom == modes.accumulation.mom0)

def mixing_ratios_remain_uniform(state):
    modes = state.prognostics.modes
    return np.all(modes.accumulation.so4 == modes.accumulation.so4[0]) and
           np.all(modes.accumulation.pom == modes.accumulation.pom[0]) and
           np.all(modes.accumulation.soa == modes.accumulation.soa[0]) and
           np.all(modes.accumulation.bc  == modes.accumulation.bc[0] ) and
           np.all(modes.accumulation.dst == modes.accumulation.dst[0]) and
           np.all(modes.accumulation.ncl == modes.accumulation.nc[0]) and
           np.all(modes.accumulation.mom == modes.accumulation.mom[0])


#------------------------
# Colun model invocation
#------------------------

column(name = 'C4_uniform_state_without_mixing',
    # time-stepping
    time = {
        'dt':    1.0,  # [s]
        'nstep': 10,
    },

    # enabled processes
    control = {
        'aerosol_cloud_interaction': {
            'droplet_activation': {
                'enable': True,
                #'supersaturation_source': 'derived_from_relative_humidity',
                #'updraft_supersaturation_closure': False,
                #'evaluation': 'diagnostic_only',
                'critical_relation': 'dilute_kappa_kohler',
                # Diameter convention: A=4*sigma*M_w/(R*T*rho_w)=2.4128472972784091e-9 m.
                # A kernel using the radius convention must use a=A/2 with matched constants.
            },
        },
    },

    # atmospheric conditions (all parameters held fixed)
    # Prescribe RH relative to liquid water and hold it fixed at each sweep point.
    # Derive fractional supersaturation s=RH-1; no prognostic vapor depletion.
    atmosphere = {
        'temperature': 273.0,                 # K; held fixed
        'pressure': 100000.0,                 # Pa; held fixed
        'initial_cloud_droplet_number': 0.0,  # m^-3
    },

    aerosols = {
        'initial_cloud_borne_number': 0.0,        # all modes, m^-3
        'initial_cloud_borne_species_mass': 0.0,  # all mode/species pairs, kg/m^3
        'modes': {
            'accumulation': {
                'number_concentration':         1.0e+8,  # m^-3
                'dry_number_median_diameter':   1.0e-7,  # m; diameter, NOT radius
                'geometric_standard_deviation': 1.8,     # dimensionless; used in initialization AND activation
                'mass_fractions': {
                    'so4': 1.0,
                },
            'aitken': {'number_concentration': 0.},
            'coarse': {'number_concentration': 0.},
            'primary_carbon': {'number_concentration': 0.},
        },
    },

    gases = {
        'so2': 0.0,
        'so4': 0.0,
        'soag': 0.0,
    },

    verification = {
        'checks': [
            mixing_ratios_remain_invariant,
            mixing_ratios_remain_uniform,
        ],
    },
)
