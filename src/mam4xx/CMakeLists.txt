include(GNUInstallDirs)

# Generate aero_config.hpp, which contains build-time parameters.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/mam4_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/mam4_config.hpp
  @ONLY
)

# Most of mam4xx is implemented in C++ headers, so we must
# install them for a client.
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/mam4_config.hpp
        aero_config.hpp
        aero_model.hpp
        aero_modes.hpp
        calcsize.hpp
        conversions.hpp
        convproc.hpp
        gasaerexch.hpp
        gasaerexch_soaexch.hpp
        gas_chem.hpp
        gas_chem_mechanism.hpp
        kerminen2002.hpp
        kohler.hpp
        mam4.hpp
        mam4_types.hpp
        merikanto2007.hpp
        nucleate_ice.hpp
        nucleation.hpp
        aging.hpp
        coagulation.hpp
        rename.hpp
        utils.hpp
        ndrop.hpp
        vehkamaki2002.hpp
        wang2008.hpp
        hetfrz.hpp
        wv_sat_methods.hpp
        drydep.hpp
        water_uptake.hpp
        wet_dep.hpp
        ndrop.hpp
        spitfire_transport.hpp
        mo_photo.hpp
        lin_strat_chem.hpp
        mo_chm_diags.hpp
        mo_setext.hpp
        mo_setsox.hpp
        mo_sethet.hpp
        tropopause.hpp
        modal_aero_opt.hpp
        aero_rad_props.hpp
        modal_aero_calcsize.hpp
        mo_setinv.hpp
        aero_model_emissions.hpp
        DESTINATION include/mam4xx)

add_library(mam4xx aero_modes.cpp aero_species.cpp gas_species.cpp)
if (MAM4XX_USE_IMPLICIT_GAS_SOLVER)
  target_compile_definitions(mam4xx PUBLIC MAM4XX_USE_IMPLICIT_GAS_SOLVER)
endif()
target_include_directories(mam4xx PUBLIC
  ${EKAT_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(mam4xx PUBLIC ekat::AllLibs)

install(TARGETS mam4xx DESTINATION ${CMAKE_INSTALL_LIBDIR})
