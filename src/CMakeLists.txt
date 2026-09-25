# MAM4xx library and friends
add_subdirectory(mam4xx)

# MAM4 process unit tests.
if (MAM4XX_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# MAM4 cross validation with Fortran implementations (double precision only)
if (MAM4XX_ENABLE_SKYWALKER AND MAM4XX_PRECISION STREQUAL "double")
  add_subdirectory(validation)
endif()

if (MAM4XX_ENABLE_BOX)
  add_subdirectory(mamboxx)
endif()

# Generate mam4xx.cmake for easy incorporation into larger projects
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/mam4xx.cmake.in
  ${PROJECT_BINARY_DIR}/share/mam4xx.cmake
  @ONLY
)
install(DIRECTORY ${PROJECT_BINARY_DIR}/share DESTINATION .)
