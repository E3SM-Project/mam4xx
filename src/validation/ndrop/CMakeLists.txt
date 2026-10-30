set(NDROP_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/ndrop)
set(NDROP_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all ndrop-related parameterizations.
add_executable(ndrop_driver
               ndrop_driver.cpp
               ccncalc.cpp
               ccncalc_single_cell.cpp
               get_activate_frac.cpp
               loadaer.cpp
               activate_modal.cpp
               explmix.cpp
               maxsat.cpp
               update_from_newcld.cpp
               update_from_cldn_profile.cpp
               update_from_explmix.cpp
               dropmixnuc.cpp
               )

target_link_libraries(ndrop_driver skywalker;validation)

# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${NDROP_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.
set(TEST_LIST
    loadaertype1_merged
    loadaertype2_merged
    loadaertype3_merged
    activate_modal_ts_1400_type1
    explmix1417_actmm_merged
    explmix1417_mm_merged
    stand_get_activate_frac_ts_1417
    activate_modal_ts_1417_type2
    stand_activate_modal_ts_1417_type2
    activate_modal_merged
    ccncalc_merged
    ccncalc_loop_ts_1417
    ccncalc_ts_1400
    maxsattype1_merged
    maxsattype2_merged
    update_from_newcld_iact0_merged
    update_from_newcld_iact1_merged
    update_from_newcld_iact2_merged
    update_from_cldn_profile_iact1_merged
    update_from_cldn_profile_iact2_merged
    stand_dropmixnuc_ts_1407
    dropmixnuc_ts_1400
    dropmixnuc_ts_1401
    dropmixnuc_ts_1402
    dropmixnuc_ts_1403
    dropmixnuc_ts_1404
    dropmixnuc_ts_1405
    dropmixnuc_ts_1406
    dropmixnuc_ts_1407
    dropmixnuc_ts_1408
    dropmixnuc_ts_1409
    dropmixnuc_ts_1410
    dropmixnuc_ts_1411
    dropmixnuc_ts_1412
    dropmixnuc_ts_1413
    dropmixnuc_ts_1414
    dropmixnuc_ts_1415
    dropmixnuc_ts_1416
    dropmixnuc_ts_1417
    update_from_explmix_merged
    )

# # matching the tests and errors, just for convenience

set(DEFAULT_TOL 1e-11)
set(DEFAULT_TOL_9 1e-09)
set(DEFAULT_TOL_DROPMIXNUC 6.8e-4)

set(ERROR_THRESHOLDS
    4e-8                      # loadaertype1_merged
    4e-8                      # loadaertype2_merged
    6e-8                      # loadaertype3_merged
    7e-5                      # activate_modal_ts_1400_type1
    2e-10                     # explmix1417_actmm_merged
    1e-9                      # explmix1417_mm_merged
    5e-5                      # stand_get_activate_frac_ts_1417
    1e-5                      # activate_modal_ts_1417_type2
    5e-6                      # stand_activate_modal_ts_1417_type2
    2e-3                      # activate_modal_merged
    2e-3                      # ccncalc_merged
    1e-3                      # ccncalc_loop_ts_1417
    5e-5                      # ccncalc_ts_1400
    ${DEFAULT_TOL_9}          # maxsattype1_merged
    ${DEFAULT_TOL_9}          # maxsattype2_merged
    2e-9                      # update_from_newcld_iact0_merged
    1e-3                      # update_from_newcld_iact1_merged
    ${DEFAULT_TOL}            # update_from_newcld_iact2_merged
    9e-3                      # update_from_cldn_profile_iact1_merged
    1e-12                     # update_from_cldn_profile_iact2_merged
    5e-5                      # stand_dropmixnuc_ts_1407
    ${DEFAULT_TOL_DROPMIXNUC} # dropmixnuc_ts_14
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    ${DEFAULT_TOL_DROPMIXNUC}
    9e-8                      # update_from_explmix_merged
   )

set(TestLabel "ndrop")

foreach(input tol IN ZIP_LISTS TEST_LIST ERROR_THRESHOLDS)
  # copy the baseline file into place.

  configure_file(
    ${NDROP_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} ndrop_driver ${NDROP_VALIDATION_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS ${TestLabel})

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${tol})
  set_property(TEST validate_${input} PROPERTY SKIP_REGULAR_EXPRESSION "regex_fail_rel_tol")
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabel})
endforeach()
