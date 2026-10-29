#include "microphysics.hpp"

#include <mam4xx/gas_chem_mechanism.hpp>
#include <mam4xx/gas_phase_chemistry.hpp>
#include <mam4xx/mo_drydep.hpp>
#include <mam4xx/mo_setinv.hpp>
#include <mam4xx/mo_sethet.hpp>
#include <mam4xx/utils.hpp>

namespace mam4::microphysics {

Microphysics::Microphysics(const Config &config): config_(config) {
  WorkArrays work_arrays = {};

  {
    const int sethet_work_len = mam4::mo_sethet::get_work_len_sethet();
    work_arrays.sethet = View1D("sethet work array", sethet_work_len);
  }

  // FIXME: initialize work arrays here!
  set_work_arrays(work_arrays, true);
}

Microphysics::Microphysics(const Config &config, const WorkArrays &work_arrays): config_(config) {
  set_work_arrays(work_arrays, false);
}

Microphysics::~Microphysics() {
  if (work_.owned) {
    // FIXME: free resources!
  }
}

void Microphysics::enable_extra_diagnostics(mam4::MicrophysDiagnosticArrays arrays) {
  extra_diagnostics_.enabled = true;
  extra_diagnostics_.arrays = arrays;
}

// Runs configured aerosol microphysics processes over a single timestep, updating the given set of
// (single column) prognostics under the given prevailing atmosphere conditions.
void Microphysics::step(const mam4::Atmosphere &atmosphere,
                        mam4::Prognostics &prognostics,
                        const ColumnLocation &location,
                        ExternalConditions &external_conditions,
                        Real dt) {
  size_t n_lev = atmosphere.temperature.size();
 
  // For guidance here, see MAMMicrophysics::run_impl in EAMxx.

  // TODO: add chlorine loading for linoz
  // TODO: add elevated emissions
  // TODO: add insolation based on orbital parameters/date and location (or just directly)
 
  // MAMMicrophysics::run_microphysics_kernels
  set_invariants(atmosphere);
  //compute_o3_column_density(atmosphere, dt);
  if constexpr (mam4::nlev > 1) {
    //photo_table(state, dt);
    sethet(atmosphere, location, external_conditions, dt);
  }
  drydep_xactive(atmosphere, location, external_conditions);
  gas_phase_chemistry(atmosphere, external_conditions, dt);
  setsox(atmosphere, external_conditions, dt);
  modal_aero_amicphys_intr(atmosphere, dt);
  vmr2mmr_cw(state, dt);
  linoz(state, dt);
}

void Microphysics::set_work_arrays(const WorkArrays &work_arrays, bool assume_ownership) {
  work_.cmfdqr = work_arrays.columns[0];

  work_.het_rates    = work_arrays.constituents[0];
  work_.qq           = work_arrays.constituents[1];
  work_.qqcw         = work_arrays.constituents[2];
  work_.vmr          = work_arrays.constituents[3];
  work_.vmr0         = work_arrays.constituents[4];
  work_.vmrcw        = work_arrays.constituents[5];
  work_.vmr_pregas   = work_arrays.constituents[6];
  work_.vmr_precld   = work_arrays.constituents[7];
  work_.dqdt_aqso4   = work_arrays.constituents[8];
  work_.dqdt_aqh2so4 = work_arrays.constituents[9];

  work_.invariants = work_arrays.fixed_species[0];

  work_.owned = assume_ownership;
}

void Microphysics::compute_diagnostics(const mam4::Atmosphere &atmosphere,
                                       const mam4::Prognostics &prognostics,
                                       mam4::Diagnostics &diagnostics) {
  diagnostics = {};
}

void Microphysics::set_external_forcing(const ColumnView &external_forcing) {
  // TODO: add when appropriate
}

void Microphysics::set_invariants(const mam4::Atmosphere &atmosphere) {
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);
  View1D cnst_offline[mam4::mo_setinv::num_tracer_cnst];
  // FIXME: populate offline constituents 
  Kokkos::parallel_for("mam4::Microphysics::set_invariants", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    mam4::mo_setinv::setinv(team,                              // in
                            work_.invariants,                  // out
                            atmosphere.temperature,            // in
                            atmosphere.vapor_mixing_ratio,     // in
                            cnst_offline,                      // in
                            atmosphere.pressure);              // in
  });
}

/*
void Microphysics::compute_o3_column_density(const Atmosphere &atmosphere,
                                             Prognostics &prognostics,
}

void Microphysics::photo_table(State &state, Real dt) {
}
*/

void Microphysics::sethet(const Atmosphere &atmosphere,
                          const ColumnLocation &location,
                          const ExternalConditions &external_conditions,
                          Real dt) {
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);
  Kokkos::parallel_for("mam4::Microphysics::sethet", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
      mam4::mo_sethet::sethet(team, atmosphere, work_.het_rates, location.latitude,
          location.surface_geopotential, work_.cmfdqr, external_conditions.prain,
          external_conditions.nevapr, dt, work_.invariants, work_.vmr,
          work_.sethet);
      });
}

void Microphysics::drydep_xactive(const Atmosphere &atmosphere, const ColumnLocation &location,
                                  ExternalConditions &external_conditions) {
  constexpr int num_gas_aerosol_constituents = mam4::gas_chemistry::gas_pcnst;
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);
  Kokkos::parallel_for("mam4::Microphysics::drydep_xactive", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
    auto work_set_het_ptr = (Real *)work_.sethet.data();
    
    // deposition flux [1/cm^2/s]
    const auto& deposition_flux = View1D(work_set_het_ptr, num_gas_aerosol_constituents);
    work_set_het_ptr += num_gas_aerosol_constituents;
    // deposition velocity [1/cm/s]
    const auto& deposition_velocity = View1D(work_set_het_ptr, num_gas_aerosol_constituents);
    work_set_het_ptr += num_gas_aerosol_constituents;

    const int surface_lev = nlev - 1; // Surface level
    // specific humidity [kg/kg]
    const Real specific_humidity = atmosphere.vapor_mixing_ratio(surface_lev);
    // surface air temperature [K]
    const Real surface_air_temperature = atmosphere.temperature(surface_lev);
    // potential temperature [K] *(temp*(1+vapor_mixing_ratio))
    //FIXME: we followed Fortran; compare with MAM4xx's potential temperature function
    const Real potential_temperature = surface_air_temperature * (1.0 + specific_humidity);
    // Surface pressure at 10m (Followed the fortran code)
    const Real pressure_10m = atmosphere.pressure(surface_lev);

    // Wind speed at the surface
    const Real wind_speed = mam4::sqrt(mam4::square(external_conditions.u_wind(surface_lev)) +
                                       mam4::square(external_conditions.v_wind(surface_lev)));
    // Total rain at the surface
    const Real rain = location.precip_liquid_surface_mass + location.precip_ice_surface_mass;

    // Downwelling solar flux at the surface (value at interface) [w/m2]
    const Real solar_flux = external_conditions.sw_solar_flux_dn(surface_lev + 1);
    const auto qq_sfc = ekat::subview(work_.qq, surface_lev);

    const mam4::seq_drydep::Data drydep_data = mam4::seq_drydep::set_gas_drydep_data();

    mam4::mo_drydep::drydep_xactive(
        drydep_data,
        location.land_use_fraction.data(),
        location.season_by_land_type.data(),
        location.surface_temperature,
        surface_air_temperature,
        potential_temperature,
        location.surface_pressure,
        pressure_10m,     // 10-meter pressure [Pa]
        specific_humidity,
        wind_speed,       // 10-meter wind speed [m/s]
        rain,             // rain content [??]
        solar_flux,       // direct shortwave surface radiation [W/m^2]
        qq_sfc.data(),    // constituent MMRs [kg/kg]
        deposition_velocity.data(),
        deposition_flux.data()
      );
      team.team_barrier();

      // Update constituent fluxes with gas drydep fluxes (dflx)
      // FIXME: Possible units mismatch (dflx is in kg/cm2/s but
      // constituent_fluxes is kg/m2/s) (Following mimics Fortran code
      // behavior but we should look into it)
      const int offset_aerosol = mam4::utils::gasses_start_ind();
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, offset_aerosol, pcnst), [&](int ispc) {
        external_conditions.constituent_fluxes(ispc) -= deposition_flux(ispc - offset_aerosol);
      });

      if (extra_diagnostics_.enabled) {
          Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, num_gas_aerosol_constituents),
            [&](const int ispc) {
              external_conditions.gas_dry_deposition_flux(ispc) = deposition_flux(ispc);
            });
      }
    });
}

void Microphysics::gas_phase_chemistry(const Atmosphere &atmosphere,
                                       const ExternalConditions &external_conditions, Real dt) {
  constexpr int num_gas_aerosol_constituents = mam4::gas_chemistry::gas_pcnst;
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);

  // Store mixing ratios before gas chemistry changes the mixing ratios.
  const auto& vmr0 = work_.vmr0;
  Kokkos::deep_copy(vmr0, work_.vmr);

  if (config_.gas_chemistry) {
    ConstituentsView gas_phase_chemistry_dvmrdt;
    if (extra_diagnostics_.enabled) {
      gas_phase_chemistry_dvmrdt = external_conditions.gas_chemistry_dvmrdt;
    }

    Kokkos::parallel_for(
        "MAMMicrophysics::run_impl::gas_phase_chemistry", policy,
        KOKKOS_LAMBDA(const ThreadTeam &team) {
        const auto photo_rates = work_.photolysis_rates;
        const auto invariants  = work_.invariants;
        const auto extfrc      = external_conditions.forcing;
        const auto het_rates   = work_.het_rates;
        const auto vmr         = work_.vmr;

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, nlev),
            [&](const int kk) {
            const auto &extfrc_k = ekat::subview(extfrc, kk);
            const auto &invariants_k = ekat::subview(invariants, kk);
            const auto &photo_rates_k = ekat::subview(photo_rates, kk);
            const auto &het_rates_k = ekat::subview(het_rates, kk);

            const Real temperature = atmosphere.temperature(kk);
            const auto &vmr_kk = ekat::subview(vmr, kk);
            mam4::microphysics::gas_phase_chemistry(
                // in
                temperature, dt, photo_rates_k.data(), extfrc_k.data(), invariants_k.data(),
                het_rates_k.data(),
                // out
                vmr_kk);
            });

        });

    if (extra_diagnostics_.enabled) {
      Kokkos::parallel_for("mam4::Microphysics::gas_phase_chemistry_dvmrdt",
                           Kokkos::RangePolicy<>(0, nlev), KOKKOS_LAMBDA(const int kk) {
        const Real pdel = atmosphere.hydrostatic_dp(kk);
        const Real mbar = mam4::Constants::molec_weight_dry_air;
        const Real gravit = mam4::Constants::gravity;
        const Real x = 1.0 / mbar * pdel / gravit;
        const auto vmr = work_.vmr;
        for (int m = 0; m < num_gas_aerosol_constituents; ++m) {
          Real adv_mass_kg_per_mole = mam4::gas_chemistry::adv_mass[m] / 1e3;
          gas_phase_chemistry_dvmrdt(m, kk) =
            x * adv_mass_kg_per_mole * (vmr(kk, m) - vmr0(kk, m)) / dt;
        }
      });
    }
  }
}

void Microphysics::setsox(const Atmosphere &atmosphere,
                          const ExternalConditions &external_conditions, Real dt) {
  constexpr int num_gas_aerosol_constituents = mam4::gas_chemistry::gas_pcnst;
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);

  const auto vmrcw = work_.vmrcw;
  const auto vmr   = work_.vmr;

  const auto vmr_pregas = work_.vmr_pregas;
  const auto vmr_precld = work_.vmr_precld;
  Kokkos::deep_copy(vmr_pregas, work_.vmr);
  Kokkos::deep_copy(vmr_precld, work_.vmrcw );

  const auto vmr_bef_aq_chem = vmr_pregas;
  const auto config_setsox = config_.setsox;
  const auto dqdt_aqso4 = work_.dqdt_aqso4;
  const auto dqdt_aqh2so4 = work_.dqdt_aqh2so4;
  const auto invariants = work_.invariants;

  const int offset_aerosol = mam4::utils::gasses_start_ind();

  const unsigned n_so4_monolayers_pcage = config_.n_so4_monolayers_pcage;

  if (config_.aqueous_chemistry) {
    Kokkos::parallel_for("Microphysics::setsox", policy, KOKKOS_LAMBDA(const ThreadTeam &team) {
        // the logic below is taken from the aero_model_gasaerexch
        // subroutine in eam/src/chemistry/modal_aero/aero_model.F90
        Kokkos::parallel_for( Kokkos::TeamVectorRange(team, nlev), [&](const int kk) {
            // extract atm state variables (input)
            Real temp = atmosphere.temperature(kk);
            Real pmid = atmosphere.pressure(kk);
            Real pdel = atmosphere.hydrostatic_dp(kk);
            Real lwc = atmosphere.liquid_mixing_ratio(kk);
            Real cldfrac = atmosphere.cloud_fraction(kk);
            Real cldnum = atmosphere.cloud_liquid_number_mixing_ratio(kk);
            const auto &invariants_k = ekat::subview(invariants, kk);

            // aqueous chemistry ...
            constexpr Real mbar = mam4::Constants::molec_weight_dry_air;
            constexpr int indexm = mam4::gas_chemistry::indexm;
            const auto dqdt_aqso4_k = ekat::subview(dqdt_aqso4, kk);
            const auto dqdt_aqh2so4_k = ekat::subview(dqdt_aqh2so4, kk);
            const auto vmrcw_k = ekat::subview(vmrcw, kk);
            const auto vmr_k = ekat::subview(vmr, kk);

            mam4::mo_setsox::setsox_single_level(
                // in
                offset_aerosol, dt, pmid, pdel, temp, mbar, lwc, cldfrac, cldnum,
                invariants_k(indexm), config_setsox,
                // out
                dqdt_aqso4_k.data(), dqdt_aqh2so4_k.data(), vmrcw_k.data(), vmr_k.data());
        });
    });

    ConstituentsView aqueous_chemistry_dvmrdt;
    if (extra_diagnostics_.enabled) {
      aqueous_chemistry_dvmrdt = external_conditions.aqueous_chemistry_dvmrdt;
    }

    if (aqueous_chemistry_dvmrdt.size()) {
      Kokkos::parallel_for("mam4::Microphysics::aqueous_chemistry_dvmrdt",
          Kokkos::RangePolicy<>(0, nlev), KOKKOS_LAMBDA(const int i) {
          const int kk   = i % nlev;
          const Real pdel = atmosphere.hydrostatic_dp(kk);
          const Real mbar = mam4::Constants::molec_weight_dry_air;
          const Real gravit = mam4::Constants::gravity;
          const Real x = 1.0 / mbar * pdel / gravit;
          for (int m = 0; m < num_gas_aerosol_constituents; ++m) {
            Real adv_mass_kg_per_mole = mam4::gas_chemistry::adv_mass[m] / 1e3;
            aqueous_chemistry_dvmrdt(m, kk) =
            x * adv_mass_kg_per_mole * (vmr(kk, m) - vmr_bef_aq_chem(kk, m)) / dt;
          }
          });
    }
  } // compute_aqueous_phase_chemistry
}

void Microphysics::modal_aero_amicphys_intr(const Atmosphere &atmosphere, Real dt) {
  auto policy = TeamPolicyFactory::get_default_team_policy(1, mam4::nlev);

  const auto wet_diameter =
    get_field_in("dgnumwet").get_view<const Real ***>();
  const auto dry_diameter =
    get_field_in("dgnum").get_view<const Real ***>();
  const auto wetdens = get_field_in("wetdens").get_view<const Real ***>();

  auto& dgncur_awet_loc = dgncur_awet_;
  auto& dgncur_a_loc = dgncur_a_;
  auto& wetdens_loc = wetdens_;
  constexpr int nmodes = mam4::AeroConfig::num_modes();
  Kokkos::parallel_for("mam4::Microphysics::modal_aero_amicphys_intr_precompute",
      Kokkos::RangePolicy<>(0, nlev), KOKKOS_LAMBDA(const int kk) {
      for (int imode = 0; imode < nmodes; imode++) {
        dgncur_awet_loc(kk, imode) = wet_diameter(imode, kk);
        dgncur_a_loc(kk, imode) = dry_diameter(imode, kk);
        wetdens_loc(kk, imode) = wetdens(imode, kk);
      }
      });

  const auto& config_amicphys = config_.amicphys;
  Kokkos::parallel_for("MAMMicrophysics::run_impl::modal_aero_amicphys_intr", policy,
      KOKKOS_LAMBDA(const ThreadTeam &team) {

      const auto& dgncur_awet_icol = ekat::subview(dgncur_awet_loc,icol);
      const auto& dgncur_a_icol = ekat::subview(dgncur_a_loc,icol);
      const auto& wetdens_icol = ekat::subview(wetdens_loc,icol);

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(team, nlev),
          [&](const int kk) {
          // calculate aerosol water content using water uptake treatment
          // * dry and wet diameters [m]
          // * wet densities [kg/m3]
          // * aerosol water mass mixing ratio [kg/kg]
          const Real temp = atmosphere.temperature(kk);
          const Real pmid = atmosphere.pressure(kk);
          const Real pdel = atmosphere.hydrostatic_dp(kk);
          const Real zm = atmosphere.height(kk);
          const Real pblh = atmosphere.planetary_boundary_layer_height;
          const Real qv = atmosphere.vapor_mixing_ratio(kk);
          const Real cldfrac = atmosphere.cloud_fraction(kk);
          const auto& dgncur_awet_kk = ekat::subview(dgncur_awet_icol,kk);
          const auto& dgncur_a_kk = ekat::subview(dgncur_a_icol,kk);
          const auto& wetdens_kk = ekat::subview(wetdens_icol,kk);

          auto vmr_kk = ekat::subview(work_.vmr,kk);
          auto vmrcw_kk = ekat::subview(work_.vmrcw,kk);
          const auto & vmr0_kk = ekat::subview(work_.vmr0,kk);
          const auto & vmr_pregas_kk = ekat::subview(work_.vmr_pregas,kk);
          const auto & vmr_precld_kk = ekat::subview(work_.vmr_precld,kk);

          // Perform aerosol microphysics (gas-aerosol exchange, nucleation, coagulation).
          mam4::microphysics::modal_aero_amicphys_intr(
              // in
              config_amicphys, dt, config_.n_so4_monolayers_pcage, temp, pmid, pdel, zm, pblh, qv,
              cldfrac,
              // out
              vmr_kk, vmrcw_kk,
              // diagnostics (out)
              kk, extra_diagnostics_.arrays.gas_aero_exchange_condensation,
              extra_diagnostics_.arrays.gas_aero_exchange_renaming,
              extra_diagnostics_.arrays.gas_aero_exchange_nucleation,
              extra_diagnostics_.arrays.gas_aero_exchange_coagulation,
              extra_diagnostics_.arrays.gas_aero_exchange_renaming_cloud_borne,
              // in
              vmr0_kk, vmr_pregas_kk, vmr_precld_kk, dgncur_a_kk, dgncur_awet_kk, wetdens_kk);
          });
      }); 
}

void Microphysics::vmr2mmr_cw(State &state, Real dt) {
}

/*
void Microphysics::linoz(State &state, Real dt) {
}
*/

} // namespace mam4::microphysics
