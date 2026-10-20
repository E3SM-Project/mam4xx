
set(CONVPROC_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/convproc)
set(CONVPROC_VALIDATION_SCRIPTS_DIR ${MAM_X_VALIDATION_DIR}/scripts)


# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.

include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all convproc-related parameterizations.
add_executable(convproc_driver convproc_driver.cpp
               update_tendency_final.cpp
               compute_column_tendency.cpp
               ma_precpevap.cpp
               ma_precpprod.cpp
               initialize_dcondt.cpp
               compute_downdraft_mixing_ratio.cpp
               ma_precpevap_convproc.cpp
               aer_vol_num_hygro.cpp
               compute_wup.cpp
               compute_massflux.cpp
               compute_ent_det_dp.cpp
               compute_midlev_height.cpp
               initialize_tmr_array.cpp
               update_qnew_ptend.cpp
               compute_wetdep_tend.cpp
               ma_activate_convproc.cpp
               compute_activation_tend.cpp
               compute_updraft_mixing_ratio.cpp
               ma_convproc_tend.cpp
               ma_convproc_dp_intr.cpp
               ma_convproc_intr.cpp
               compute_tendencies.cpp
               ma_resuspend_convproc.cpp)
target_link_libraries(convproc_driver skywalker;validation)


# Copy some Python scripts from mam_x_validation to our binary directory.
foreach(script
        compare_mam4xx_mam4.py)
  configure_file(
    ${CONVPROC_VALIDATION_SCRIPTS_DIR}/${script}
    ${CMAKE_CURRENT_BINARY_DIR}/${script}
    COPYONLY
  )
endforeach()

# Run the driver in several configurations to produce datasets.

set(TestLabel "convproc")

# convproc
foreach (input
         update_tendency_final
         compute_column_tendency
         ma_precpevap
         ma_precpprod_ts_355
         initialize_dcondt
         compute_downdraft_mixing_ratio
         ma_precpevap_convproc
         ma_resuspend_convproc
         aer_vol_num_hygro
         compute_wup
         compute_massflux
         compute_massflux_small
         compute_ent_det_dp
         compute_midlev_height
         initialize_tmr_array_1
         initialize_tmr_array_2
         update_qnew_ptend_true
         update_qnew_ptend_false
         compute_wetdep_tend
         ma_activate_convproc_54
         ma_activate_convproc_68
         compute_activation_tend_do_act_true
         compute_activation_tend_do_act_false
         compute_updraft_mixing_ratio
         ma_convproc_tend
         ma_convproc_dp_intr
         ma_convproc_intr_ts_355
         ma_convproc_intr_only_aero_ts_355
         compute_tendencies
         )

  # copy the baseline file into place; is the skywalker file produced by fortran code?
  configure_file(
    ${CONVPROC_VALIDATION_DIR}/mam_${input}.py
    ${CMAKE_CURRENT_BINARY_DIR}/mam_${input}.py
    COPYONLY
  )

  # add a test to run the skywalker driver
  add_test(run_${input} convproc_driver ${CONVPROC_VALIDATION_DIR}/${input}.yaml)
  set_tests_properties(run_${input} PROPERTIES LABELS ${TestLabel})

  # add a test to validate mam4xx's results against the baseline.
  # Select a threshold error slightly bigger than the largest relative error for the threshold error.
  # Conservative approach: At the time of this commit, threshold error=2e-6.
  # compare_mam4xx_mam4.py <module1.py> <module2.py> <check_norms> <threshold error>
  if (${input} STREQUAL compute_activation_tend_do_act_true)
    set(TOL 6e-6)
  elseif (${input} STREQUAL ma_activate_convproc_54)
    set(TOL 6e-5)
  elseif (${input} STREQUAL ma_convproc_tend)
    set(TOL 1e-4)
  elseif (${input} STREQUAL ma_convproc_intr_ts_355)
    set(TOL 1e-4)
  elseif (${input} STREQUAL compute_updraft_mixing_ratio)
    set(TOL 2e-4)
  else ()
    set(TOL 2e-6)
  endif ()
  add_test(validate_${input} python3 compare_mam4xx_mam4.py mam4xx_${input}.py mam_${input}.py True ${TOL})
  set_property(TEST validate_${input} PROPERTY SKIP_REGULAR_EXPRESSION "regex_fail_rel_tol")
  set_tests_properties(validate_${input} PROPERTIES DEPENDS run_${input})
  set_tests_properties(validate_${input} PROPERTIES LABELS ${TestLabel})
endforeach()
