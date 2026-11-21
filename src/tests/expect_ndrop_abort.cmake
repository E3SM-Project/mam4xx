# Run intentionally fatal input/error tests in isolated processes. A nonzero
# exit alone is insufficient: require the specific kernel error as well.
execute_process(
  COMMAND "${TEST_EXE}" implicit_input_errors_abort -c "${SECTION}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  TIMEOUT 60)
set(output "${output}\n${error}")
if("${result}" STREQUAL "0" OR result MATCHES "[Tt]imeout" OR
   NOT output MATCHES "${EXPECTED_MESSAGE}" OR
   output MATCHES "invalid input must stop")
  message(FATAL_ERROR "Expected ndrop kernel error '${EXPECTED_MESSAGE}', got '${result}':\n${output}")
endif()
