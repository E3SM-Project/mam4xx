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

## TODO

[X] Initialize qq, qqcw, stateq, vmr (etc) work arrays in an EAMxx-sensible fashion
[ ] Implement standalone work array initialization
[X] Organize microphysics into a self-contained interface
[X] Determine "test problems" for box model
[ ] Determine box model output shape
[ ] Start carving out an ACI interface
