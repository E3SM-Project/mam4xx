# This folder contains Skywalker inputs and Python scripts for processing outputs.
set(MAM_X_VALIDATION_DIR ${mam_x_validation_SOURCE_DIR})

# This library has some common utilities used by Skywalker drivers in this
# directory tree, including the functions in testing.cpp for creating column
# views, atmospheric state, prognostics, and diagnostics used by unit tests.
add_library(validation validation.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../tests/testing.cpp)
target_link_libraries(validation PUBLIC mam4xx skywalker)
target_include_directories(validation PUBLIC
  ${skywalker_SOURCE_DIR}/include
  ${skywalker_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../tests)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#--------------------------
# MAM4xx Skywalker drivers
#--------------------------
add_subdirectory(nucleation)
add_subdirectory(calcsize)
add_subdirectory(convproc)
add_subdirectory(coagulation)
add_subdirectory(aging)
add_subdirectory(rename)
add_subdirectory(nucleate_ice)
add_subdirectory(hetfrz)
add_subdirectory(aero_model)
add_subdirectory(wetdep)
add_subdirectory(drydep)
add_subdirectory(gas_chem)
add_subdirectory(ndrop)
add_subdirectory(water_uptake)
add_subdirectory(mo_photo)
add_subdirectory(lin_strat_chem)
add_subdirectory(mo_drydep)
add_subdirectory(mo_setext)
add_subdirectory(mo_sethet)
add_subdirectory(mo_setsox)
add_subdirectory(tropopause)
add_subdirectory(aerosol_optics)
add_subdirectory(mo_setinv)
add_subdirectory(aero_emissions)
add_subdirectory(micro_gasaerexch)
add_subdirectory(tracer_data)
add_subdirectory(modal_aero_amicphys_subareas)
