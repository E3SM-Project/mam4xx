message(STATUS "Configuring mam4xx box model")

# Configure an appropriate Python interpreter for parsing input and running verification checks.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

#-------------------------------------------------------------
# Single Column model (Same # of vertical columns as library)
#-------------------------------------------------------------

add_executable(mamscm
  column.cpp
  config_py.cpp
  main.cpp
  run.cpp
  microphysics.cpp
  aerosol_cloud_interaction.cpp
)
target_link_libraries(mamscm PUBLIC mam4xx Python3::Python)
target_include_directories(mamscm PUBLIC
  ${EKAT_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

#-------------------------------
# Box model (1 vertical column)
#-------------------------------

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/box)
set(MAM4XX_NUM_VERTICAL_LEVELS 1)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../mam4xx/mam4_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/box/mam4_config.hpp
  @ONLY
)

add_executable(mamboxx
  column.cpp
  config_py.cpp
  main.cpp
  run.cpp
  microphysics.cpp
  aerosol_cloud_interaction.cpp
)
target_link_libraries(mamboxx PUBLIC mam4xx Python3::Python)
target_include_directories(mamboxx PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/box>
  ${EKAT_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
