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

* **How are `qq` and `qqcw` constituent arrays populated?** See
  `mam4::utils::extract_xxx_from_prognostics`.

## TODO

[X] Initialize qq, qqcw, stateq, vmr (etc) work arrays in an EAMxx-sensible fashion
[ ] Implement standalone work array initialization
[X] Organize microphysics into a self-contained interface
[ ] Figure out how MAM gets wet/dry diameters and wet density fields
[ ] Determine one or more "test problems" for aerosol microphysics (benchmark against F90 box model?)
[ ] Determine box model output shape
