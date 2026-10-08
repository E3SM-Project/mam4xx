message(STATUS "Configuring mam4xx box model")

add_executable(mamboxx main.cpp config.cpp run.cpp)
target_link_libraries(mamboxx PUBLIC mam4xx)
target_include_directories(mamboxx PUBLIC
  ${EKAT_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
