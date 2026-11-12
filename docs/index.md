# Overview

MAM4xx is a performance-portable, GPU-capable C++ implementation of the 4-mode
version of the Modal Aerosol Model (MAM), intended for use with E3SM.

* The [Installation](installation.md) guide shows you how to build and install
  MAM4xx on your own machine or on a supported high-performance platform.
* The [Theory Guide](theory/index.md) explains the scientific formulations,
  assumptions, and process coupling represented in MAM4xx, beginning with
  [sulfur chemistry](theory/sulfur-chemistry.md).
* The [Developer](development.md) guide lays out some basic principles and
  guidelines we use in developing MAM4xx.
* In the [Testing](testing.md) guide, we describe in general terms the
  methodology we use to test MAM4xx and its aerosol processes.

### Acknowledgements

MAM4xx was developed by an interdisciplinary team consisting of aerosol and
atmospheric researchers, applied mathematicians, and software engineers. It was
created for the [EAGLES project](https://climatemodeling.science.energy.gov/projects/enabling-aerosol-cloud-interactions-global-convection-permitting-scales-eagles),
an effort to improve the treatment of aerosols in
[E3SM](https://climatemodeling.science.energy.gov/projects/energy-exascale-earth-system-model),
the Department of Energy's global climate model. The source code is available on
[GitHub](https://github.com/E3SM-Project/mam4xx). This effort was funded by
the Office of Science's [Biological and Environmental
Research](https://science.osti.gov/ber) Program.
