set(AERO_MODEL_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/aero_model)
set(AERO_MODEL_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all aero_model-related parameterizations.
add_executable(aero_model_driver  aero_model_driver.cpp
               modal_aero_bcscavcoef_init.cpp
              calc_1_impact_rate.cpp
              modal_aero_bcscavcoef_get.cpp
              aero_model_wetdep.cpp
              aero_model_wetdep_convproc.cpp
              aero_model_calcsize_water_uptake_dr.cpp)
target_link_libraries(aero_model_driver skywalker;validation)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${AERO_MODEL_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    calc_1_impact_rate_ts_0
    modal_aero_bcscavcoef_get_ts_355
    modal_aero_bcscavcoef_init_ts_0
    aero_model_wetdep_ts_379
    stand_aero_model_calcsize_water_uptake_dr_ts_379
    baseline_aero_model_wetdep_ts_379
    aero_model_wetdep_convproc_ts_3
    aero_model_wetdep_convproc_no_gas_no_aerosol_ts_3
    )

set(DEFAULT_TOL_BASELINE 1e-13)

set(ERROR_THRESHOLDS  
   5e-8                    # calc_1_impact_rate_ts_0
   1e-9                   # modal_aero_bcscavcoef_get_ts_355
   2e-4                    # modal_aero_bcscavcoef_init_ts_0
   2e-5                    # aero_model_wetdep_ts_379
   1e-6                    # stand_aero_model_calcsize_water_uptake_dr_ts_379
   2e-5                    # baseline_aero_model_wetdep_ts_379
   1e-4 #aero_model_wetdep_convproc_ts_3
   1e-5 #aero_model_wetdep_convproc_no_gas_no_aerosol_ts_3
   )

set(TestLabel "aero_model")

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${AERO_MODEL_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} aero_model_driver ${AERO_MODEL_VALIDATION_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS ${TestLabel})

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_property(TEST validate_${input} PROPERTY SKIP_REGULAR_EXPRESSION "regex_fail_rel_tol")
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabel})

endforeach()
