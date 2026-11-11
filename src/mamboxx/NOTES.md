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

The number of possible checks that we can do with a column/box model is too large to keep in
compiled code. I think a better approach (that I've used very effectively in a SciDAC project) is
to use something like [muparser](https://beltoforion.de/en/muparser/index.php) to parse mathematical
expressions in a `checks` section. `muparser` in particular includes various operators that reduce
these expressions to `true`/`false` values, so we can indicate a successful check with `true` and
an unsuccessful one with `false`.

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
- initial_interstitial_number_and_species_mass
    - initial_cloud_borne_number_and_species_mass
    - critical_dry_diameter
    - activated_number_fraction
    - activated_dry_mass_fraction
    - activated_number_concentration       # f_N*N_total, m^-3
    - activated_species_mass_concentration # f_M*M_j,total, kg/m^3
    - clipping_and_floor_events

See also `output.fields` in the example script for test problem **B1**.

### Referring to a quantity at a specific vertical level in columns

Here we have to be a little cute, since `muparser` doesn't support arrays. But actually,
[muparserx](https://beltoforion.de/en/muparserx/index.php) does! So this seems to be a non-issue.

### Experiments/property tests involving parameter sweeps

We can devise many more property tests if the box/column model driver incorporates some basic
perturbed-parameter features. For example, Susannah's example input for problem **B1** includes a
sweep over various relative humidities. We can construct expressions for e.g. sensitivities using
vector or matrix notation. We can also add some simple functions that provide more clarity to
expressions.

## TODO

[X] Initialize qq, qqcw, stateq, vmr (etc) work arrays in an EAMxx-sensible fashion
[ ] Implement standalone work array initialization
[X] Organize microphysics into a self-contained interface
[X] Determine "test problems" for box model
[ ] Determine box model output shape
[ ] Start carving out an ACI interface
