include(EkatCreateUnitTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mam4_test_config.hpp.in
  ${PROJECT_BINARY_DIR}/include/mam4_test_config.hpp
  @ONLY
)

add_library(mam4xx_tests kohler_verification.cpp
                         atmosphere_utils.cpp
                         testing.cpp)
target_include_directories(mam4xx_tests PUBLIC ${PROJECT_BINARY_DIR}/include)
target_link_libraries(mam4xx_tests mam4xx)

#------------
# Unit Tests
#------------

# We use the EkatCreateUnitTest CMake function to create unit tests that are
# configured properly to work in the Kokkos environment provided by EKAT.
# EkatCreateUnitTest must be called with the following options:
# 1. LIBS mam4xx_tests <-- links against mam4xx
# 2. USER_DEFINED_TEST_SESSION <-- uses mam4xx's setup/breakdown functions

EkatCreateUnitTest(utils_unit_tests utils_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_gasaerexch_unit_tests mam4_gasaerexch_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_coagulation_unit_tests mam4_coagulation_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_convproc_unit_tests mam4_convproc_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_rename_unit_tests mam4_rename_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_aging_unit_tests mam4_aging_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
# EkatCreateUnitTest(mam4_amicphys_1gridcell_tests mam4_amicphys_1gridcell.cpp
#   LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_nucleate_ice_unit_tests mam4_nucleate_ice_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_wet_deposition_unit_tests mam4_wet_deposition_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_dry_deposition_unit_tests mam4_dry_deposition_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_spitfire_transport_unit_tests mam4_spitfire_transport_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_mo_setsox_unit_tests mam4_mo_setsox_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
EkatCreateUnitTest(mam4_aero_emissions_unit_tests mam4_aero_emissions_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)

EkatCreateUnitTest(mam4_photo_setcol mam4_photo_setcol.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)

EkatCreateUnitTest(gas_chem_direct_solver_unit_tests
  gas_chem_direct_solver_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)

if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel") #The Intel compiler always emit a warning for inline functions.
  target_compile_options(utils_unit_tests PRIVATE)
  target_compile_options(mam4_gasaerexch_unit_tests PRIVATE)
  target_compile_options(mam4_coagulation_unit_tests PRIVATE)
  target_compile_options(mam4_convproc_unit_tests PRIVATE)
  # target_compile_options(mam4_amicphys_1gridcell_tests PRIVATE)
  target_compile_options(mam4_rename_unit_tests PRIVATE)
  target_compile_options(mam4_aging_unit_tests PRIVATE)
  target_compile_options(mam4_nucleate_ice_unit_tests PRIVATE)
  target_compile_options(mam4_spitfire_transport_unit_tests PRIVATE)
  target_compile_options(mam4_mo_setsox_unit_tests PRIVATE)
  target_compile_options(mam4_aero_emissions_unit_tests PRIVATE)
endif ()

EkatCreateUnitTest(mam4_ndrop_unit_tests mam4_ndrop_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)

EkatCreateUnitTest(aero_modes_unit_tests aero_modes_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)

EkatCreateUnitTest(aero_config_unit_tests aero_config_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)

EkatCreateUnitTest(conversions_unit_tests conversions_unit_tests.cpp
  LIBS mam4xx_tests USER_DEFINED_TEST_SESSION)
