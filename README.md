# MAM4xx

[![Build Status](https://github.com/E3SM-Project/mam4xx/workflows/auto_test/badge.svg)](https://github.com/E3SM-Project/mam4xx/actions)
[![Code Coverage](https://codecov.io/gh/E3SM-Project/mam4xx/branch/main/graph/badge.svg?token=OI33WNBS7N)](https://codecov.io/gh/E3SM-Project/mam4xx)

This repository contains the source code for a performance-portable C++
implementation of the MAM4 modal aerosol model (with 4 fixed modes).

## Required Software

To build MAM4xx, you need:

* [CMake v3.17+](https://cmake.org/)
* GNU Make
* reliable C and C++ compilers
* optionally, a working MPI installation (like [OpenMPI](https://www.open-mpi.org/)
  or [MPICH](https://www.mpich.org/)), if you want to use MAM4xx in a
  multi-node parallel environment
* optionally, a GPU with appropriate drivers installed

You can obtain all of these freely on the Linux and Mac platforms. On Linux,
just use your favorite package manager. On a Mac, you can get the Clang C/C++
compiler by installing XCode, and then use a package manager like
[Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/) to get the
rest.

For example, to download the relevant software on your Mac using Homebrew, type

```
brew install cmake openmpi
```

## Configuring MAM4xx

From the top-level source directory, invoke CMake with appropriate options like this:

```
cmake -S . -B <build-directory> [OPTIONS]
```

Available options are:

* `MAM4XX_PRECISION`: Set this to `single` or `double` to set floating point precision
  (default: `double`).
* `MAM4XX_ENABLE_GPU`: Set this flag to enable MAM4xx to run on GPUs. It should be accompanied by
  a `MAM4XX_DEVICE_ARCH` setting with an architecture flag that conforms to those listed
  [here](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#gpu-architectures).
  Specifically, MAM4xx sets `Kokkos_ARCH_${MAM4XX_DEVICE_ARCH}` to configure an appropriate backend.
  **This parameter is only required for standalone MAM4xx configurations.**
* `MAM4XX_ENABLE_COVERAGE`: Set this flag to have MAM4xx generate code coverage reports. This
  requires [lcov](https://github.com/linux-test-project/lcov) to be installed.
* `MAM4XX_ENABLE_TESTS`: Set this flag to enable MAM4xx's unit tests.
* `MAM4XX_ENABLE_SKYWALKER`: Set this flag to enable MAM4xx's cross validation testing against
  MAM4 Fortran datasets generated during the porting process.
* `MAM4XX_NUM_VERTICAL_LEVELS`: Set this to the number of vertical levels in an atmospheric
  column (default: 72).

Here's an example that configures MAMxx to run on CPUs with cross validation and testing enabled:

```
cmake -S . -B build \
  -DMAM4XX_PRECISION=double \
  -DMAM4XX_ENABLE_SKYWALKER=ON
```

Here's a GPU configuration example that uses NVidia's `AMPERE86` architecture:

```
cmake -S . -B build \
  -DMAM4XX_ENABLE_GPU=ON \
  -DMAM4XX_DEVICE_ARCH=AMPERE86 \
  -DMAM4XX_PRECISION=double
```

By default, CMake 3.x generates Makefiles, and CMake 4.x generates Ninja files. You can set this
manually with the -G flag.

You can also specify a location to install MAM4xx with `CMAKE_INSTALL_PREFIX`.

### `build.sh` script

You can also use `build.sh` (located in this directory) to configure and build mam4xx all at once.
Type `./build.sh` for usage information. Some simple examples:

* Debuggable double-precision CPU build:
  `./build.sh build-cpu cpu double Debug`
* Debuggable double-precision NVidia GPU build (AMPERE86 architecture):
  `./build.sh build-cuda gpu double Debug nvidia AMPERE86`

## Building MAM4xx

To build or rebuild MAM4xx:

1. Change to your build directory (`build` in the example above) and type `make` (or `ninja`).
2. To run tests for the library (and the driver, if configured), type
   `make test` or `ninja test`.
3. To install the model to the location indicated by `PREFIX` in your
   `config.sh` script, type `make install` or `ninja install`. By default, products are installed
   in `include`, `lib`, `bin`, and `share` subdirectories within your build
   directory.

You can skip this step if you just ran `build.sh`.

## Tools and Workflows

### Checking C++ formatting, and auto-formatting

Our C++ style rules are described in the [MAM4xx developer guide](https://github.com/E3SM-Project/mam4xx/blob/main/docs/development.md).
We enforce them using `clang-format`. If you have the correct version of
`clang-format` installed, you can use the following targets to check and fix
all C++ code in the `src` subdirectory:

* `make format-cxx-check`: checks C++ formatting in all source files and reports
  any non-conforming code
* `make format-cxx`: applies C++ formatting rules to all source files, editing
  them in place. Try to do this in a separate commit from your other work.

You can run either of these targets from your build directory. If you have a
different version of `clang-format` than the one we support, you'll get an error
message telling you the correct version to install when you use either of these
targets.

### Analyzing code coverage

You can get a code coverage report if you've enabled mam4xx to build with
code coverage instrumentation. This option is configurable in your `config.sh`
script if you uncomment the `COVERAGE=ON` line, or if you run CMake directly
with the `-DENABLE_COVERAGE=ON` flag. You must have the
[LCOV](https://lcov.readthedocs.io/en/latest/index.html) tool
installed to generate reports.

To generate a code coverage report:

1. Build mam4xx with `make -j`
2. Run the unit tests and validation tests with `make test`
3. Generate the coverage report with `make coverage`

You will see a file named `coverage.info` in your build directory. This can
be used with LCOV to visualize source files and functions that do and don't
have coverage.

Our automated testing system generates code coverage reports and uploads them
to [codecov.io](https://about.codecov.io/) so they appear in the message feed
for relevant pull requests.

### Continuous Integration

See [the SNL CI README](.github/workflows/README.md) for more detailed information.

### Generating Documentation

Documentation for MAM4xx can be built using
[`mkdocs`](https://squidfunk.github.io/mkdocs-material/).
In order to build and view the
documentation, you must download `mkdocs` and its Material theme:

```pip3 install mkdocs mkdocs-material```

Then, run `mkdocs serve` from the root directory of your MAM4xx repo,
and point your browser to [`http://localhost:8000`](http://localhost:8000).

You can also view the [MAM4xx developer guide](https://github.com/E3SM-Project/mam4xx/blob/main/docs/development.md)
on GitHub.

## Disclaimer

This material was prepared as an account of work sponsored by an agency
of the United States Government. Neither the United States Government
nor the United States Department of Energy, nor Battelle, nor any of
their employees, nor any jurisdiction or organization that has
cooperated in the development of these materials, makes any warranty,
express or implied, or assumes any legal liability or responsibility for
the accuracy, completeness, or usefulness or any information, apparatus,
product, software, or process disclosed, or represents that its use
would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service
by trade name, trademark, manufacturer, or otherwise does not
necessarily constitute or imply its endorsement, recommendation, or
favoring by the United States Government or any agency thereof, or
Battelle Memorial Institute. The views and opinions of authors expressed
herein do not necessarily state or reflect those of the United States
Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY 

*operated by* BATTELLE

*for the* UNITED STATES DEPARTMENT OF ENERGY

*under Contract DE-AC05-76RL01830*




