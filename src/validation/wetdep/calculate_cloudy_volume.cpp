// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <validation.hpp>
#include <vector>

using namespace skywalker;

void test_calculate_cloudy_volume_process(const Input &input, Output &output) {
  const int nlev = 72;
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt", "ncol", "is_tot_cld", "cld", "lprec"};

  // Iterate over input_variables and error if not in input
  for (std::string name : input_variables) {
    if (!input.has(name.c_str()) && !input.has_array(name.c_str())) {
      std::cerr << "Required name for variable: " << name << std::endl;
      exit(1);
    }
  }

  // Parse input
  // These first two values are unused
  EKAT_ASSERT(0 == input.get("dt"));
  EKAT_ASSERT(4 == input.get("ncol"));
  const int is_tot_cld = input.get("is_tot_cld");
  const std::vector<Real> cld = input.get_array("cld");
  const std::vector<Real> lprec = input.get_array("lprec");
  EKAT_ASSERT(nlev == cld.size());
  EKAT_ASSERT(nlev == lprec.size());

  auto cld_dev = mam4::validation::create_column_view(nlev);
  auto lprec_dev = mam4::validation::create_column_view(nlev);
  auto cldv_dev = mam4::validation::create_column_view(nlev);
  auto sumppr_all_dev = mam4::validation::create_column_view(nlev);
  {
    auto cld_host = Kokkos::create_mirror_view(cld_dev);
    for (int i = 0; i < nlev; ++i)
      cld_host[i] = cld[i];
    Kokkos::deep_copy(cld_dev, cld_host);
  }
  {
    auto lprec_host = Kokkos::create_mirror_view(lprec_dev);
    for (int i = 0; i < nlev; ++i)
      lprec_host[i] = lprec[i];
    Kokkos::deep_copy(lprec_dev, lprec_host);
  }
  Kokkos::parallel_for(
      "wetdep::calculate_cloudy_volume", 1, KOKKOS_LAMBDA(const int) {
        auto lprec = [&](int i) { return lprec_dev[i]; };
        mam4::wetdep::calculate_cloudy_volume(nlev, cld_dev, lprec, is_tot_cld,
                                              cldv_dev);

        sumppr_all_dev[0] = lprec_dev[0];
        for (int i = 1; i < nlev; i++)
          sumppr_all_dev[i] = sumppr_all_dev[i - 1] + lprec_dev[i];
      });

  // Create mirror views for output arrays
  {
    std::vector<Real> cldv(72);
    auto vals = Kokkos::create_mirror_view(cldv_dev);
    Kokkos::deep_copy(vals, cldv_dev);
    for (int i = 0; i < nlev; ++i)
      cldv[i] = vals[i];
    output.set("cldv", cldv);
  }
  {
    std::vector<Real> sumppr_all(72);
    auto vals = Kokkos::create_mirror_view(sumppr_all_dev);
    Kokkos::deep_copy(vals, sumppr_all_dev);
    for (int i = 0; i < nlev; ++i)
      sumppr_all[i] = vals[i];
    output.set("sumppr_all", sumppr_all);
  }
}

void test_calculate_cloudy_volume(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_calculate_cloudy_volume_process(input, output);
  });
}
