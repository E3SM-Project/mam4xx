// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <validation.hpp>
#include <vector>

using namespace skywalker;

void test_wetdep_clddiag_process(const Input &input, Output &output) {
  using View1DHost = mam4::HostType::view_1d<Real>;
  using View1D = mam4::ndrop::View1D;
  // pver is constant and the size of our arrays
  const int pver = 72;
  int nlev = 72;
  Real pblh = 1000;
  auto atm = mam4::testing::create_atmosphere(nlev, pblh);
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt"};

  std::string input_arrays[] = {"ncol",   "temperature", "pmid", "pdel",
                                "cmfdqr", "evapc",       "cldt", "cldcu",
                                "cldst",  "evapr",       "prain"};

  // Iterate over input_variables and error if not in input
  for (std::string name : input_variables) {
    if (!input.has(name.c_str())) {
      std::cerr << "Required name for variable: " << name << std::endl;
      exit(1);
    }
  }
  // Iterate over input_arrays and error if not in input
  for (std::string name : input_arrays) {
    if (!input.has_array(name.c_str())) {
      std::cerr << "Required name for array: " << name << std::endl;
      exit(1);
    }
  }

  // Parse input
  // These first two values are unused
  // auto dt = input.get("dt");
  // auto ncol = input.get_array("ncol");
  auto temperature = input.get_array("temperature");
  auto pmid = input.get_array("pmid");
  auto pdel = input.get_array("pdel");
  auto cmfdqr = input.get_array("cmfdqr");
  auto evapc = input.get_array("evapc");
  auto cldt = input.get_array("cldt");
  auto cldcu = input.get_array("cldcu");
  auto cldst = input.get_array("cldst");
  auto evapr = input.get_array("evapr");
  auto prain = input.get_array("prain");

  // Assert arrays are the correct size
  EKAT_ASSERT(temperature.size() == pver);
  EKAT_ASSERT(pmid.size() == pver);
  EKAT_ASSERT(pdel.size() == pver);
  EKAT_ASSERT(cmfdqr.size() == pver);
  EKAT_ASSERT(evapc.size() == pver);
  EKAT_ASSERT(cldt.size() == pver);
  EKAT_ASSERT(cldcu.size() == pver);
  EKAT_ASSERT(cldst.size() == pver);
  EKAT_ASSERT(evapr.size() == pver);
  EKAT_ASSERT(prain.size() == pver);

  auto view = [](std::string n, auto v) {
    View1DHost host(v.data(), pver);
    View1D dev(n, pver);
    Kokkos::deep_copy(dev, host);
    return dev;
  };
  View1D temperature_view = view("temperature", temperature);
  View1D pmid_view = view("pmid", pmid);
  View1D pdel_view = view("pdel", pdel);
  View1D cmfdqr_view = view("cmfdqr", cmfdqr);
  View1D evapc_view = view("evapc", evapc);
  View1D cldt_view = view("cldt", cldt);
  View1D cldcu_view = view("cldcu", cldcu);
  View1D cldst_view = view("cldst", cldst);
  View1D evapr_view = view("evapr", evapr);
  View1D prain_view = view("prain", prain);

  // Prepare device views for output arrays
  auto cldv_dev = mam4::validation::create_column_view(pver);
  auto cldvcu_dev = mam4::validation::create_column_view(pver);
  auto cldvst_dev = mam4::validation::create_column_view(pver);
  auto rain_dev = mam4::validation::create_column_view(pver);

  Kokkos::parallel_for(
      "wetdep::clddiag", 1, KOKKOS_LAMBDA(const int) {
        mam4::wetdep::clddiag(pver, temperature_view, pmid_view, pdel_view,
                              cmfdqr_view, evapc_view, cldt_view, cldcu_view,
                              cldst_view, evapr_view, prain_view, cldv_dev,
                              cldvcu_dev, cldvst_dev, rain_dev);
      });

  // Create mirror views for output arrays
  auto cldv_host = Kokkos::create_mirror_view(cldv_dev);
  auto cldvcu_host = Kokkos::create_mirror_view(cldvcu_dev);
  auto cldvst_host = Kokkos::create_mirror_view(cldvst_dev);
  auto rain_host = Kokkos::create_mirror_view(rain_dev);

  // Copy values back to the host
  Kokkos::deep_copy(cldv_host, cldv_dev);
  Kokkos::deep_copy(cldvcu_host, cldvcu_dev);
  Kokkos::deep_copy(cldvst_host, cldvst_dev);
  Kokkos::deep_copy(rain_host, rain_dev);

  // Create Vectors for output arrays and copy in place
  std::vector<Real> cldv(cldv_host.data(), cldv_host.data() + pver);
  std::vector<Real> cldvcu(cldvcu_host.data(), cldvcu_host.data() + pver);
  std::vector<Real> cldvst(cldvst_host.data(), cldvst_host.data() + pver);
  std::vector<Real> rain(rain_host.data(), rain_host.data() + pver);

  // Set the output values
  output.set("cldv", cldv);
  output.set("cldvcu", cldvcu);
  output.set("cldvst", cldvst);
  output.set("rain", rain);
}

void test_wetdep_clddiag(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdep_clddiag_process(input, output);
  });
}
