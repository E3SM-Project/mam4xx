# A direct CTest WILL_FAIL test still fails on signals. Inspect a separately
# launched process and require both abnormal termination and the solver message.
execute_process(COMMAND "${EXECUTABLE}" "${CASE}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
if ("${result}" STREQUAL "77")
  message("SKIP: the configured wrapper returns diagnostics instead of aborting")
  return()
endif()
if (NOT "${result}" MATCHES "[Aa]bort" OR
    NOT "${output}${error}" MATCHES "${EXPECTED_MESSAGE}")
  message(FATAL_ERROR "Expected solver abort for ${CASE}; result: ${result}\n${output}${error}")
endif()
