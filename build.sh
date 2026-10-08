#!/usr/bin/env bash

# This script builds mam4xx in standalone mode. Run it like this:
#
# `./build.sh <build_dir> <device> <precision> <build_type> [gpu_type] [gpu_arch]`
#
# where
# * <build_dir> is the build directory (e.g. `build`) in which you would like
#   mam4xx configured and built
# * <device> (either `cpu` or `gpu`), identifies the device type for which mam4xx
#   is built.
# * <precision> (either `single` or `double`) determines the precision of
#   floating point numbers used in mam4xx. Default: double
# * <build_type> (either `Debug` or `Release`) determines whether mam4xx is built
#   optimized or for debugging. Default: Debug
# OPTIONAL Args (REQUIRED for GPU builds)
# * [gpu_type] (`nvidia`, `amd`, `intel`) this is the "brand" of GPU
# * [gpu_arch] (e.g., HOPPER90, AMD_GFX90A, INTEL_PVC) this is the "model" of
#              GPU, indicating the generation, compute capability, etc.
#   * The full list of options may be found on the kokkos docs site:
# https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#architectures

BUILD_DIR=$1
DEVICE=$2
PRECISION=$3
BUILD_TYPE=$4
GPU_TYPE=$5
DEVICE_ARCH=$6

if [[ ! -x `which ninja` ]]; then
  echo "Couldn't find the ninja build tool. This script uses Ninja instead of Make."
  echo "Please install ninja using your favorite package manager."
  exit
fi

if [[ "$BUILD_DIR" == "" ]]; then
  echo "mam4xx build directory was not specified!"
  echo "Usage: $0 <build_dir> <device> <precision> <build_type> [gpu_type] [gpu_arch]"
  exit
fi

# Set defaults.
if [[ "$DEVICE" == "" ]]; then
  DEVICE=cpu
  echo "No device specified. Selected cpu."
fi
if [[ "$PRECISION" == "" ]]; then
  PRECISION=double
  echo "No floating point precision specified. Selected double."
fi
if [[ "$BUILD_TYPE" == "" ]]; then
  BUILD_TYPE=Debug
  echo "No build type specified. Selected Debug."
fi

# Validate options
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
  echo "Invalid device specified: $DEVICE"
  exit
fi
if [[ "$PRECISION" != "single" && "$PRECISION" != "double" ]]; then
  echo "Invalid precision specified: $PRECISION (must be single or double)"
  exit
fi
# This should format the string as required (first letter capitalized)
BUILD_TYPE="${BUILD_TYPE^}"
if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
  echo "Invalid optimization specified: $BUILD_TYPE"
  echo "Must be Debug or Release (case-sensitive)"
  exit
fi

# ==============================================================================
# below are the default compiler choices for gpu builds and are set based on
# the command line args.
# ==============================================================================
if [[ "$DEVICE" == "gpu" ]]; then
  ENABLE_GPU=ON
  if [[ "$GPU_TYPE" == "" ]]; then
    echo "ERROR: GPU_TYPE not provided for GPU build."
    echo "Must be nvidia, amd, or intel (case-sensitive)"
    exit
  elif [[ "$GPU_TYPE" != "nvidia" && "$GPU_TYPE" != "amd" && "$GPU_TYPE" != "intel" ]]; then
    echo "Device provided as 'gpu', but invalid GPU_TYPE specified: ${GPU_TYPE}."
    echo "Must be nvidia, amd, or intel (case-sensitive)"
    exit
  fi
  if [[ "$DEVICE_ARCH" == "" ]]; then
    echo "ERROR: DEVICE_ARCH not provided for GPU build."
    exit
  fi

  if [[ "$GPU_TYPE" == "nvidia" ]]; then
    echo "GPU_TYPE nvidia (${DEVICE_ARCH})--setting default compilers."
    CXX=g++
    CC=gcc
  elif [[ "$GPU_TYPE" == "amd" ]]; then
    echo "GPU_TYPE amd--setting default compilers."
    CXX=hipcc
    CC=amdclang
  elif [[ "$GPU_TYPE" == "intel" ]]; then
    echo "GPU_TYPE intel--setting default compilers."
    CXX=icpx
    CC=icx
  fi
else
  ENABLE_GPU=OFF
fi

if [[ "$DEVICE" == "cpu" ]]; then
  # Default cpu compilers (can be overridden by environment variables)
  echo "Setting compilers for CPU device (override with CC and CXX)."
  if [[ -z $CC ]]; then
    CC=gcc
  fi
  if [[ -z $CXX ]]; then
    CXX=g++
  fi
fi

echo "C compiler:   ${CC}"
echo "C++ compiler: ${CXX}"

# ==============================================================================
#     DON'T CHANGE ANYTHING BELOW HERE UNLESS YOU KNOW WHAT YOU'RE DOING
# ==============================================================================

echo "Configuring mam4xx in $BUILD_DIR."
rm -rf .build
cmake -S . -B $BUILD_DIR \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DMAM4XX_ENABLE_GPU=$ENABLE_GPU \
  -DMAM4XX_PRECISION=$PRECISION \
  -DMAM4XX_DEVICE_ARCH=$DEVICE_ARCH \
  -DMAM4XX_ENABLE_BOXMODEL=ON \
  -G Ninja \
  || exit

echo "Building mam4xx in $BUILD_DIR..."
cmake --build $BUILD_DIR
