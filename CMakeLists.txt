cmake_minimum_required(VERSION 3.17.0)

#----------------------
# Configurable options
#----------------------

option(MAM4XX_ENABLE_GPU       "Enable GPU support" OFF)
option(MAM4XX_ENABLE_COVERAGE  "Enable code coverage instrumentation" OFF)
option(MAM4XX_ENABLE_SKYWALKER "Enable Skywalker cross validation" ON)
option(MAM4XX_ENABLE_TESTS     "Enable unit tests" ON)
option(MAM4XX_EVAL_DIAGNOSTICS "Enable evaluation diagnostics" OFF)

# Single or double precision?
if (CMAKE_VERSION GREATER_EQUAL 4.2)
  set(CACHE{MAM4XX_PRECISION}   TYPE STRING HELP "Precision of Real numbers (single, [double])" VALUE double)
  set(CACHE{MAM4XX_DEVICE_ARCH} TYPE STRING HELP "Kokkos-sensible GPU device architecture"      VALUE "")
else()
  set(MAM4XX_PRECISION   double CACHE STRING "Precision of Real numbers (single, [double])")
  set(MAM4XX_DEVICE_ARCH ""     CACHE STRING "Kokkos-sensible GPU device architecture")
endif()

# are we building as part of EAMxx, or in standalone mode?
if (NOT DEFINED EAMXX_VERSION_MAJOR)
  set(MAM4XX_STANDALONE ON)
  set(CXX_STANDARD 17)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # emit compile_commands.json for use with clangd
else()
  set(MAM4XX_STANDALONE OFF)
endif()

if( NOT DEFINED SCREAM_NUM_VERTICAL_LEV)
  set(MAM4XX_NUM_VERTICAL_LEVELS 72 CACHE STRING "the number of vertical levels per column")
else()
  # use the same number of levels as EAMxx if MAM4xx is run under EAMxx (or SCREAM)
  set(MAM4XX_NUM_VERTICAL_LEVELS ${SCREAM_NUM_VERTICAL_LEV} CACHE STRING "the number of vertical levels per column equals scream num levels")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#--------------------------
# End configurable options
#--------------------------

project(mam4xx)
enable_language(CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "C++ compiler is ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "C compiler is ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")

# Version numbers.
set(MAM4XX_MAJOR_VERSION 0)
set(MAM4XX_MINOR_VERSION 1)
set(MAM4XX_PATCH_VERSION 0)

set(MAM4XX_VERSION "${MAM4XX_MAJOR_VERSION}.${MAM4XX_MINOR_VERSION}.${MAM4XX_PATCH_VERSION}")
message(STATUS "Configuring MAM4xx v${MAM4XX_VERSION} (build type: ${CMAKE_BUILD_TYPE})")

if(MAM4XX_ENABLE_GPU)
  if (MAM4XX_STANDALONE)
    if (NOT MAM4XX_DEVICE_ARCH)
      message(FATAL_ERROR "MAM4XX_ENABLE_GPU is set but no MAM4XX_DEVICE_ARCH is given")
    endif()
    message(STATUS "Configuring for GPU (${MAM4XX_DEVICE_ARCH})")
  else()
    message(STATUS "Configuring for GPU")
  endif()
else() #CPU
  message(STATUS "Configuring for CPU")
endif()

# Precision of floating point numbers.
if (MAM4XX_PRECISION STREQUAL "double")
  set(MAM4XX_REAL_TYPE double)
  set(MAM4XX_DOUBLE_PRECISION 1)
elseif (MAM4XX_PRECISION STREQUAL "single")
  set(MAM4XX_REAL_TYPE float)
  set(MAM4XX_DOUBLE_PRECISION 0)
else()
  message(FATAL_ERROR "Invalid MAM4XX_PRECISION: ${MAM4XX_PRECISION} (must be single or double)")
endif()
message(STATUS "Using ${MAM4XX_PRECISION} precision floating point numbers")

message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

include_directories(${PROJECT_BINARY_DIR}/src/mam4xx)
include_directories(${PROJECT_SOURCE_DIR}/src)

#-----------------------
# Third-party libraries
#-----------------------
include(FetchContent)

# for standalone mode, configure EKAT and Kokkos
if (MAM4XX_STANDALONE)
  set(EKAT_ENABLE_ALL_PACKAGES ON)
  set(EKAT_ENABLE_MPI OFF) # standalone mam4xx is a column model, no need for MPI
  set(EKAT_ENABLE_TESTS OFF)
  set(EKAT_SKIP_FIND_YAML_CPP ON) # use yaml-cpp from submodule
  # HACK: We need this for our CI builds because EKAT uses SSH to get its submodules, so we
  # HACK: have a GitHub Action do it for us. We can remove this if/when EKAT switches to
  # HACK: CMake's FetchContent machinery like we have.
  if (DEFINED ENV{EKAT_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_EKAT $ENV{EKAT_SOURCE_DIR})
    set(ekat_SOURCE_DIR $ENV{EKAT_SOURCE_DIR})
    message(STATUS "Building EKAT from source at $ENV{EKAT_SOURCE_DIR}")
  endif()

  # Fetch a specific SHA for ekat so we can update manually when needed and not
  # every time ekat updates the master branch.  
  # NOTE: This same tag is used in the github workflow actions and should match
  # the tags in 
  #      .github/workflows/at2_gcc-cuda.yml 
  # and 
  #      .github/workflows/gh_gcc-cpu.yml
  # files for the actions/checkout@v4 ref: option. 
  FetchContent_Declare(
    EKAT
    GIT_REPOSITORY https://github.com/E3SM-Project/EKAT.git
    GIT_TAG        007bdcff6cb26948517d2d26f3d7ca005363d119 # 2026/04/13
  )
  list(APPEND mam4xx_tpls EKAT)

  # set up Kokkos backends as needed
  set(Kokkos_ENABLE_LIBDL OFF CACHE BOOL "Whether to enable the LIBDL library")
  set(Kokkos_ENABLE_DEPRECATED_CODE_4 OFF CACHE BOOL "Whether code deprecated in major release 4 is available")
  if (CMAKE_BUILD_TYPE STREQUAL Release)
    set(Kokkos_ENABLE_DEBUG FALSE)
    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON)
  else()
    set(Kokkos_ENABLE_DEBUG TRUE)
    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION OFF)
  endif()
  if (MAM4XX_ENABLE_GPU)
    set(Kokkos_ENABLE_SERIAL ON) # provides host
    set(Kokkos_ARCH_${MAM4XX_DEVICE_ARCH} ON)
    if ("${MAM4XX_DEVICE_ARCH}" MATCHES "AMD")
      set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable HIP Kokkos backend")
    elseif ("${MAM4XX_DEVICE_ARCH}" MATCHES "INTEL")
      set(Kokkos_ENABLE_SYCL ON CACHE BOOL "Enable SYCL Kokkos backend")
    else()
      set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA Kokkos backend")
      set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA lambdas")
    endif()
  else() # CPU
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
  endif()
endif()

if (MAM4XX_ENABLE_SKYWALKER)
  # build Skywalker
  set(SKYWALKER_PRECISION ${MAM4XX_PRECISION})
  FetchContent_Declare(
    skywalker
    GIT_REPOSITORY https://github.com/E3SM-Project/skywalker.git
    GIT_TAG        aa5f192690be6f8ca657b40b4895b1b5e43ec063 # 2026/09/18
  )
  list(APPEND mam4xx_tpls skywalker)

  # bring in MAM4 Fortran validation datasets
  FetchContent_Declare(
    mam_x_validation
    GIT_REPOSITORY https://github.com/E3SM-Project/mam_x_validation.git
    GIT_TAG        8abea036f30f1d4c3005b205169bd2d4c0a701c3 # 2026/04/23
  )
  list(APPEND mam4xx_tpls mam_x_validation)
endif()

# build the third-party libraries and make their targets available
FetchContent_MakeAvailable(${mam4xx_tpls})

# finish configuring EKAT for standalone builds
if (MAM4XX_STANDALONE)
  set(EKAT_INCLUDE_DIRS
    ${ekat_SOURCE_DIR}/src/kokkos
    ${ekat_SOURCE_DIR}/extern/kokkos/core/src
    ${ekat_SOURCE_DIR}/extern/Catch2/single_include
  )
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ekat_SOURCE_DIR}/cmake/")
  if (MAM4XX_ENABLE_GPU)
    if ("${MAM4XX_DEVICE_ARCH}" MATCHES "AMD")
      set(CMAKE_CXX_COMPILER "hipcc")
    elseif ("${MAM4XX_DEVICE_ARCH}" MATCHES "INTEL")
      set(CMAKE_CXX_COMPILER "icpx")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -nocudainc")
      link_libraries(sycl)
    else() # Nvidia
      file(COPY ${ekat_SOURCE_DIR}/extern/kokkos/bin/nvcc_wrapper
           DESTINATION ${CMAKE_BINARY_DIR}/bin)
      set(EKAT_NVCC_WRAPPER ${CMAKE_BINARY_DIR}/bin/nvcc_wrapper)
      set(CMAKE_CXX_COMPILER ${EKAT_NVCC_WRAPPER})
      # the esa_on_defaulted_function_ignored supresses a warning about KOKKOS markups of defaulted constructors/destructors
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKokkos_CXX_STANDARD=C++17")
  endif()
endif()

#---------------------------
# End third-party libraries
#---------------------------

# code coverage
if (MAM4XX_ENABLE_COVERAGE)
  message(STATUS "Enabling code coverage instrumentation")
  find_program(LCOV_EXE lcov DOC "Lcov code coverage tool")
  if (LCOV_EXE MATCHES "NOTFOUND")
    message(FATAL_ERROR "Could not find lcov for code coverage reporting!")
  endif()

  # Add code coverage compiler/linker flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

  # Add a "make coverage" target.
  add_custom_target(coverage ${LCOV_EXE} --capture --directory . -o unfiltered_coverage.info
    COMMAND ${LCOV_EXE} --remove unfiltered_coverage.info -o coverage.info '*/external/*' '*/tests/*' '*/validation/*'
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Generating coverage report (coverage.info)")
endif()

# support for valgrind (standalone, Linux only)
if (MAM4XX_STANDALONE AND NOT TARGET memcheck)
  include(add_memcheck_target)
  add_memcheck_target()
endif()

# testing
if (MAM4XX_ENABLE_TESTS OR MAM4XX_ENABLE_SKYWALKER)
  include(CTest)
  enable_testing()
endif()

# source directories
add_subdirectory(src)

# formatting targets (using clang-format):
# * `format-cxx` formats the code in place
# * `format-cxx-check` reports whether the code is properly formatted
include(add_formatting_targets)
add_formatting_targets()
