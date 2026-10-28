set(WETDEP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/wetdep)
set(WETDEP_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all wetdep-related parameterizations.
add_executable(wetdep_driver
               wetdep_driver.cpp
               wetdep_clddiag.cpp
               wetdep_resusp_nonlinear.cpp
               wetdep_resusp_noprecip.cpp
               wetdep_scavenging.cpp
               compute_evap_frac.cpp
               rain_mix_ratio.cpp
               calculate_cloudy_volume.cpp
               local_precip_production.cpp
               wetdep_resusp.cpp
               wetdepa_v2.cpp)
target_link_libraries(wetdep_driver skywalker;validation)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${WETDEP_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    clddiag
    wetdep_resusp_nonlinear_130
    wetdep_resusp_nonlinear_230
    wetdep_resusp_noprecip_130
    wetdep_resusp_noprecip_230
    wetdep_scavenging_true
    wetdep_scavenging_false
    compute_evap_frac
    rain_mix_ratio
    calculate_cloudy_volume
    local_precip_production
    wetdep_resusp_130
    wetdep_resusp_230
    wetdepa_v2_0
    wetdepa_v2_130
    wetdepa_v2_230
    )

set(DEFAULT_TOL 1e-6)

set(ERROR_THRESHOLDS
    ${DEFAULT_TOL} # clddiag
    ${DEFAULT_TOL} # wetdep_resusp_nonlinear_130
    ${DEFAULT_TOL} # wetdep_resusp_nonlinear_230
    ${DEFAULT_TOL} # wetdep_resusp_noprecip_130
    ${DEFAULT_TOL} # wetdep_resusp_noprecip_230
    ${DEFAULT_TOL} # wetdep_scavenging_true
    ${DEFAULT_TOL} # wetdep_scavenging_false
    ${DEFAULT_TOL} # compute_evap_frac
    ${DEFAULT_TOL} # rain_mix_ratio
    ${DEFAULT_TOL} # calculate_cloudy_volume
    ${DEFAULT_TOL} # local_precip_production
    ${DEFAULT_TOL} # wetdep_resusp_130
    ${DEFAULT_TOL} # wetdep_resusp_230
    ${DEFAULT_TOL} # wetdepa_v2_0
    ${DEFAULT_TOL} # wetdepa_v2_130
    ${DEFAULT_TOL} # wetdepa_v2_230
    )

set(TestLabel "wetdep")

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${WETDEP_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} wetdep_driver ${WETDEP_VALIDATION_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS ${TestLabel})

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_property(TEST validate_${input} PROPERTY SKIP_REGULAR_EXPRESSION "regex_fail_rel_tol")
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabel})

endforeach()
