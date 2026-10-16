// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <catch2/catch.hpp>

#include <mam4xx/aero_model.hpp>
#include <mam4xx/convproc.hpp>
#include <validation.hpp>

using namespace skywalker;

namespace {

using View1D = typename mam4::DeviceType::view_1d<Real>;
using View2D = typename mam4::DeviceType::view_2d<Real>;
using View1DHost = typename mam4::HostType::view_1d<Real>;

void get_input(const Input &input, const std::string &name, const int size,
               std::vector<Real> &host, View1D &dev) {
  host = input.get_array(name);
  dev = View1D(name, size);

  EKAT_ASSERT(host.size() == size);
  auto host_view = View1DHost((Real *)host.data(), size);
  Kokkos::deep_copy(dev, host_view);
}

void set_output(Output &output, const std::string &name, const int size,
                std::vector<Real> &host, const View1D &dev) {
  host.resize(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0; i < size; ++i) {
    host[i] = host_view[i];
  }
  output.set(name, host);
}

void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, std::vector<Real> &host,
               View2D &dev) {
  host = input.get_array(name);
  dev =View2D(name,rows,cols);                                           
  EKAT_ASSERT(host.size() == rows * cols);
  mam4::validation::convert_1d_vector_to_2d_view_device(host, dev);
}

template <int COLS>
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Kokkos::View<Real *[COLS]> &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  // Column Major layout (Fortran)
  for (int j = 0, n = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i, ++n) {
      host[n] = host_view(i, j);
    }
  output.set(name, host);
}

void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host, const View2D &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  // Column Major layout (Fortran)
  for (int j = 0, n = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i, ++n) {
      host[n] = host_view(i, j);
    }
  output.set(name, host);
}

} // namespace

void ma_convproc_intr(Ensemble *ensemble) {
  // Create AeroConfig with default aero species (matches eamxx pattern)
  const mam4::AeroConfig aero_config;
  const auto aero_species = aero_config.aero_species;

  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    const int pcnst = mam4::aero_model::pcnst;

    // Read scalar inputs
    const bool convproc_do_aer = std::round(input.get_array("convproc_do_aer")[0]);
    const bool convproc_do_gas = std::round(input.get_array("convproc_do_gas")[0]);
    // dt is an array with one element for single column
    const auto dt_array = input.get_array("dt");
    const Real dt = dt_array[0];
    // jt and maxg are arrays with one element for single column
    const auto jt_array = input.get_array("jt");
    const auto maxg_array = input.get_array("maxg");
    const int ktop = Kokkos::round(jt_array[0] - 1); // Convert to C++ index (jt is 1-indexed Fortran)
    const int kbot = Kokkos::round(maxg_array[0]); 

    // Prepare resuspension array
    int mmtoo_prevap_resusp[pcnst];
    {
      if (input.has("mmtoo_prevap_resusp")) {
        const auto resusp_db = input.get_array("mmtoo_prevap_resusp");
        for (int i = 0; i < pcnst; ++i)
          mmtoo_prevap_resusp[i] = resusp_db[i] - 1; // Convert to C++ index
      } else {
        // Default: no resuspension mapping (use -1 for all species)
        for (int i = 0; i < pcnst; ++i)
          mmtoo_prevap_resusp[i] = -1;
      }
    }

    // Prepare species_class array
    int species_class[pcnst];
    {
      const auto species_class_db = input.get_array("species_class");
      for (int i = 0; i < pcnst; ++i)
        species_class[i] = species_class_db[i];
    }

    // Read 1D input arrays
    std::vector<Real> temperature_host, pmid_host, dpdry_host, dp_frac_host,
        icwmrdp_host, rprddp_host, evapcdp_host, dlf_host, du_host, eu_host,
        ed_host, dp_host;
    View1D temperature_dev, pmid_dev, dpdry_dev, dp_frac_dev,
        icwmrdp_dev, rprddp_dev, evapcdp_dev, dlf_dev, du_dev, eu_dev, ed_dev,
        dp_dev;

    get_input(input, "state_t", nlev, temperature_host, temperature_dev);
    get_input(input, "state_pmid", nlev, pmid_host, pmid_dev);
    get_input(input, "state_pdeldry", nlev, dpdry_host, dpdry_dev);
    get_input(input, "dp_frac", nlev, dp_frac_host, dp_frac_dev);
    get_input(input, "icwmrdp", nlev, icwmrdp_host, icwmrdp_dev);
    get_input(input, "rprddp", nlev, rprddp_host, rprddp_dev);
    get_input(input, "evapcdp", nlev, evapcdp_host, evapcdp_dev);
    get_input(input, "dlf", nlev, dlf_host, dlf_dev);
    get_input(input, "du", nlev, du_host, du_dev);
    get_input(input, "eu", nlev, eu_host, eu_dev);
    get_input(input, "ed", nlev, ed_host, ed_dev);
    get_input(input, "dp", nlev, dp_host, dp_dev);

    // Read 2D input/output arrays (state_q and ptend_q)
    std::vector<Real> state_q_host, ptend_q_host;
    View2D state_q_dev, ptend_q_dev;

    get_input(input, "state_q", nlev, pcnst, state_q_host, state_q_dev);

    // Allocate output array for ptend_q
    View1D ptend_q_col_view("ptend_q", nlev * pcnst);
    ptend_q_dev = View2D(ptend_q_col_view.data(), nlev, pcnst);

    // Initialize ptend_q to zero
    Kokkos::deep_copy(ptend_q_dev, 0.0);

    // Allocate and initialize output arrays
    std::vector<Real> aerdepwetis_host;
    View1D aerdepwetis_dev;
    get_input(input, "aerdepwetis_init", pcnst, aerdepwetis_host,
              aerdepwetis_dev);

    Kokkos::View<bool *> ptend_lq_dev("ptend_lq", pcnst);
    Kokkos::deep_copy(ptend_lq_dev, false);

    // Allocate work array for scratch1Dviews
    const int work_size = mam4::convproc::get_scratch1d_work_len(nlev);

    View1D work_view("work", work_size);
    Real *work_ptr = work_view.data();

    // Scratch arrays - create outside parallel_for
    Kokkos::View<Real *>
        scratch1Dviews[mam4::ConvProc::Col1DViewInd::NumScratch];
    mam4::convproc::init_scratch_1d_views(scratch1Dviews, work_ptr, nlev);

    // Execute the function
    auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool ptend_lq[pcnst];
          Real aerdepwetis[pcnst];

          // Initialize output arrays based on species_class and flags
          // ptend_lq[i] is true if species i should have a tendency:
          //   - aerosol species when convproc_do_aer is true
          //   - gas species when convproc_do_gas is true
          for (int i = 0; i < pcnst; ++i) {
            ptend_lq[i] =
                (species_class[i] == mam4::ConvProc::species_class::aerosol &&
                 convproc_do_aer) ||
                (species_class[i] == mam4::ConvProc::species_class::gas &&
                 convproc_do_gas);
            aerdepwetis[i] = aerdepwetis_dev[i];
          }

          // Call the function
          mam4::convproc::ma_convproc_intr(
              team, aero_species, scratch1Dviews, convproc_do_aer,
              convproc_do_gas, nlev, temperature_dev.data(), pmid_dev.data(),
              dpdry_dev.data(), dt, dp_frac_dev.data(), icwmrdp_dev.data(),
              rprddp_dev.data(), evapcdp_dev.data(), dlf_dev.data(),
              du_dev.data(), eu_dev.data(), ed_dev.data(), dp_dev.data(), ktop,
              kbot, species_class, mmtoo_prevap_resusp, state_q_dev,
              ptend_q_dev, ptend_lq, aerdepwetis);

          // Copy results back
          for (int i = 0; i < pcnst; ++i) {
            ptend_lq_dev[i] = ptend_lq[i];
            aerdepwetis_dev[i] = aerdepwetis[i];
          }
        });

    // Set outputs
    set_output(output, "ptend_q", nlev, pcnst, ptend_q_host, ptend_q_dev);

    // Output ptend_lq (convert bool to Real for output)
    std::vector<Real> ptend_lq_host(pcnst);
    {
      auto host_view = Kokkos::create_mirror_view(ptend_lq_dev);
      Kokkos::deep_copy(host_view, ptend_lq_dev);
      for (int i = 0; i < pcnst; ++i)
        ptend_lq_host[i] = host_view[i] ? 1.0 : 0.0;
    }
    output.set("ptend_lq", ptend_lq_host);

    // Output aerdepwetis
    set_output(output, "aerdepwetis", pcnst, aerdepwetis_host,
               aerdepwetis_dev);
  });
}
