# MAM box model NOTES

## Terminology

### Symbols
 
| Symbol      | Description                 | Nominal Value | Declared in file         |
| ----------- | --------------------------- | ------------- | ------------------------ |
| `gas_pcnst` | number of gas phase species | 31            | `gas_chem_mechanism.hpp` |
| `pcnst`     | total number of tracers     | 40            | `mam4_config.hpp.in`     |
| `nfs`       | number of "fixed" species   | 8             | `gas_chem_mechanism.hpp` |

### Containers

| Container   | Description                                                       |
| ----------- | ----------------------------------------------------------------- |
| `q`         | interstitial chemical constitents work array (including aerosols) |
| `qqcw`      | cloudborne chemical constituents work array (including aerosols)  |

Note that `gas_pcnst` is not the number of unique gas species. That is 13.

## Questions

* The dry and wet geometric mean diameters and the wet density are computed by the wet scavenging
  eamxx/mam process, but not stored within mam4xx data structures. Should we find a place to store
  these things in mam4xx's prognostic variables, or should we treat them as diagnosed quantities and
  compute them whenever we need them?

## Verification and Property Testing

We are experimenting with interpreted languages (Python, Lua) to allow the use of callable functions
in input files. See the contents of the `examples/` directory for such examples.

In order to implement such a mechanism, we need a way of addressing all relevant quantities in the
aerosol state. Here are some suggestions (provided that we can have variable names containing dots):

* `constants.<constant>`: any physical constant used by mam4xx
* `<mode>.number_fraction`: the number fraction for all aerosol particles in the given mode
* `<mode>.<species>.mass_fraction`: the mass fraction for the given aerosol species in the given mode
* `atmosphere.supersaturation`
* `atmosphere.relative_humidity`
* `atmosphere.temperature`
* `atmosphere.pressure`
* `<mode>.dry_number_median_diameter`
* `<mode>.geometric_std_dev`
* `kappa`: ??? 

Other things we may want to support (from Susannah's example input):
* initial_interstitial_number_and_species_mass
* initial_cloud_borne_number_and_species_mass
* critical_dry_diameter
* activated_number_fraction
* activated_dry_mass_fraction
* activated_number_concentration
* activated_species_mass_concentration
* clipping_and_floor_events

Some of these could be implemented as callable functions themselves.

### Experiments/property tests involving parameter sweeps

We can devise many more property tests if the box/column model driver incorporates some basic
perturbed-parameter features. For example, Susannah's example input for problem **B1** includes a
sweep over various relative humidities. We can construct expressions for e.g. sensitivities using
vector or matrix notation in Python or Lua. We can also add some simple functions that provide more
clarity to expressions.

## Code Structure

Ultimately, here's what I'd like to have at the top level:

```
* namespace mam4::aerosol_cloud_interactions
    * class DropletActivation
    * class HeterogeneousFreezing
* namespace mam4::constituent_fluxes
    * class ...
* namespace mam4::dry_deposition
* namespace mam4::microphysics
    * class Microphysics
* namespace mam4::optics
    * class ...
* namespace mam4::emissions
    * class ...
* namespace mam4::wet_scavenging
    * class ...
```

The classes would call all the stuff that's currently in mam4xx. The source code can be reorganized
into namespace-oriented folders in `src/mam4xx`, e.g. `src/mam4xx/aci`, `src/mam4xx/microphysics`,
etc.

## TODO

[X] Initialize qq, qqcw, stateq, vmr (etc) work arrays in an EAMxx-sensible fashion
[ ] Implement standalone work array initialization
[X] Organize microphysics into a class
[ ] Organize dropmixnuc into class
[X] Determine "test problems" for box model
[ ] Implement a Python input parser with verification checking
[ ] Determine box model output shape
[ ] Start carving out an ACI interface
