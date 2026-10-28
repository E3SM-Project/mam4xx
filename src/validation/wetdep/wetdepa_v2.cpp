// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>

#include <validation.hpp>
#include <vector>

using namespace skywalker;

KOKKOS_INLINE_FUNCTION
void wetdep_prevap_step(
    const int is_st_cu, const int mam_prevap_resusp_optcc, const Real pdel_ik,
    const Real pprdx, const Real srcx, const Real arainx,
    const Real precabx_old, const Real precabx_base_old,
    const Real scavabx_old, const Real precnumx_base_old,
    Real &precabx_new, Real &precabx_base_new,
    Real &scavabx_new, Real &precnumx_base_new) {
  constexpr Real small_value_30 = 1.e-30;
  const Real gravit = mam4::Constants::gravity;

  scavabx_new = scavabx_old;
  precnumx_base_new = precnumx_base_old;

  Real tmpa = mam4::max(0.0, pprdx * pdel_ik / gravit);
  precabx_base_new = mam4::max(0.0, precabx_base_old + tmpa);
  precabx_new = mam4::utils::min_max_bound(0.0, precabx_base_new, precabx_old + tmpa);

  if (mam_prevap_resusp_optcc <= 130) {
    tmpa = mam4::max(0.0, srcx * pdel_ik / gravit);
    scavabx_new = mam4::max(0.0, scavabx_old + tmpa);
  } else {
    if (precabx_base_new < small_value_30) {
      precnumx_base_new = 0.0;
    } else if (precabx_base_new > precabx_base_old) {
      tmpa = arainx * mam4::wetdep::flux_precnum_vs_flux_prec_mpln(
                          precabx_base_new / arainx, is_st_cu);
      precnumx_base_new = mam4::max(0.0, tmpa);
    } else {
      precnumx_base_new = precnumx_base_old;
    }
  }
}

void test_wetdepa_v2_process(const Input &input, Output &output) {
  const int nlev = 72;
  // Ensemble parameters
  // Declare array of strings for input names
  std::string input_variables[] = {"dt",
                                   "ncol",
                                   "deltat",
                                   "pdel",
                                   "cmfdqr",
                                   "evapc",
                                   "dlf",
                                   "conicw",
                                   "precs",
                                   "evaps",
                                   "cwat",
                                   "cldt",
                                   "cldc",
                                   "cldvcu",
                                   "cldvst",
                                   "tracer",
                                   "mam_prevap_resusp_optcc",
                                   "is_strat_cloudborne",
                                   "scavcoef",
                                   "f_act_conv",
                                   "qqcw",
                                   "sol_factb",
                                   "sol_facti",
                                   "sol_factic"};

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
  const int is_strat_cloudborne = input.get("is_strat_cloudborne");
  const int mam_prevap_resusp_optcc = input.get("mam_prevap_resusp_optcc");
  const Real deltat = input.get("deltat");
  const Real sol_factb = input.get("sol_factb");
  const Real sol_facti = input.get("sol_facti");

  const std::vector<Real> pdel = input.get_array("pdel");
  const std::vector<Real> cmfdqr = input.get_array("cmfdqr");
  const std::vector<Real> evapc = input.get_array("evapc");
  const std::vector<Real> dlf = input.get_array("dlf");
  const std::vector<Real> conicw = input.get_array("conicw");
  const std::vector<Real> precs = input.get_array("precs");
  const std::vector<Real> evaps = input.get_array("evaps");
  const std::vector<Real> cwat = input.get_array("cwat");
  const std::vector<Real> cldt = input.get_array("cldt");
  const std::vector<Real> cldc = input.get_array("cldc");
  const std::vector<Real> cldvcu = input.get_array("cldvcu");
  const std::vector<Real> cldvst = input.get_array("cldvst");
  const std::vector<Real> tracer = input.get_array("tracer");
  const std::vector<Real> scavcoef = input.get_array("scavcoef");
  const std::vector<Real> f_act_conv = input.get_array("f_act_conv");
  const std::vector<Real> qqcw = input.get_array("qqcw");
  const std::vector<Real> sol_factic = input.get_array("sol_factic");

  auto pdel_dev = mam4::validation::create_column_view(nlev);
  auto cmfdqr_dev = mam4::validation::create_column_view(nlev);
  auto evapc_dev = mam4::validation::create_column_view(nlev);
  auto dlf_dev = mam4::validation::create_column_view(nlev);
  auto conicw_dev = mam4::validation::create_column_view(nlev);
  auto precs_dev = mam4::validation::create_column_view(nlev);
  auto evaps_dev = mam4::validation::create_column_view(nlev);
  auto cwat_dev = mam4::validation::create_column_view(nlev);
  auto cldt_dev = mam4::validation::create_column_view(nlev);
  auto cldc_dev = mam4::validation::create_column_view(nlev);
  auto cldvcu_dev = mam4::validation::create_column_view(nlev);
  auto cldvst_dev = mam4::validation::create_column_view(nlev);
  auto tracer_dev = mam4::validation::create_column_view(nlev);
  auto scavcoef_dev = mam4::validation::create_column_view(nlev);
  auto f_act_conv_dev = mam4::validation::create_column_view(nlev);
  auto qqcw_dev = mam4::validation::create_column_view(nlev);
  auto sol_factic_dev = mam4::validation::create_column_view(nlev);

  auto copy_to_dev = [](mam4::ColumnView dev, std::vector<Real> vec) {
    auto host = Kokkos::create_mirror_view(dev);
    for (int i = 0; i < nlev; ++i)
      host[i] = vec[i];
    Kokkos::deep_copy(dev, host);
  };
  copy_to_dev(pdel_dev, pdel);

  copy_to_dev(pdel_dev, pdel);
  copy_to_dev(cmfdqr_dev, cmfdqr);
  copy_to_dev(evapc_dev, evapc);
  copy_to_dev(dlf_dev, dlf);
  copy_to_dev(conicw_dev, conicw);
  copy_to_dev(precs_dev, precs);
  copy_to_dev(evaps_dev, evaps);
  copy_to_dev(cwat_dev, cwat);
  copy_to_dev(cldt_dev, cldt);
  copy_to_dev(cldc_dev, cldc);
  copy_to_dev(cldvcu_dev, cldvcu);
  copy_to_dev(cldvst_dev, cldvst);
  copy_to_dev(tracer_dev, tracer);
  copy_to_dev(scavcoef_dev, scavcoef);
  copy_to_dev(f_act_conv_dev, f_act_conv);
  copy_to_dev(qqcw_dev, qqcw);
  copy_to_dev(sol_factic_dev, sol_factic);

  auto fracis_dev = mam4::validation::create_column_view(nlev);
  auto scavt_dev = mam4::validation::create_column_view(nlev);
  auto iscavt_dev = mam4::validation::create_column_view(nlev);
  auto icscavt_dev = mam4::validation::create_column_view(nlev);
  auto isscavt_dev = mam4::validation::create_column_view(nlev);
  auto bcscavt_dev = mam4::validation::create_column_view(nlev);
  auto bsscavt_dev = mam4::validation::create_column_view(nlev);
  auto rcscavt_dev = mam4::validation::create_column_view(nlev);
  auto rsscavt_dev = mam4::validation::create_column_view(nlev);

  Kokkos::parallel_for(
      "wetdep::wetdepa_v2", 1, KOKKOS_LAMBDA(const int) {
        constexpr Real small_value_2 = 1.e-2;
        constexpr Real small_value_12 = 1.e-12;
        constexpr Real small_value_36 = 1.e-36;
        const Real omsm = 1.0 - 2 * mam4::epsilon();
        const Real gravit = mam4::Constants::gravity;

        Real precabs = 0;
        Real precabc = 0;
        Real scavabs = 0;
        Real scavabc = 0;
        Real precabs_base = 0;
        Real precabc_base = 0;
        Real precnums_base = 0;
        Real precnumc_base = 0;

        for (int kk = 0; kk < nlev; ++kk) {
          // Call single-level mam4::wetdep::wetdepa_v2
          mam4::wetdep::wetdepa_v2(
              deltat, pdel_dev[kk], cmfdqr_dev[kk], evapc_dev[kk], dlf_dev[kk],
              conicw_dev[kk], precs_dev[kk], evaps_dev[kk], cwat_dev[kk],
              cldt_dev[kk], cldc_dev[kk], cldvcu_dev[kk], cldvst_dev[kk],
              sol_factb, sol_facti, sol_factic_dev[kk], mam_prevap_resusp_optcc,
              is_strat_cloudborne, scavcoef_dev[kk], f_act_conv_dev[kk],
              tracer_dev[kk], qqcw_dev[kk], precabs, precabc, scavabs, scavabc,
              precabs_base, precabc_base, precnums_base, precnumc_base,
              scavt_dev[kk], bcscavt_dev[kk], rcscavt_dev[kk], rsscavt_dev[kk]);

          // Evaporation fraction
          const Real fracev_st = mam_prevap_resusp_optcc
                                     ? mam4::wetdep::compute_evap_frac(
                                           pdel_dev[kk], evaps_dev[kk], precabs)
                                     : 0;
          const Real fracev_cu = mam_prevap_resusp_optcc
                                     ? mam4::wetdep::compute_evap_frac(
                                           pdel_dev[kk], evapc_dev[kk], precabc)
                                     : 0;

          // Scavenging quantities
          const Real clddiff = cldt_dev[kk] - cldc_dev[kk];
          const Real tracer_tmp = mam4::min(
              qqcw_dev[kk], tracer_dev[kk] * (clddiff / mam4::max(small_value_2,
                                                                 (1. - clddiff))));
          const Real tracer_incu =
              f_act_conv_dev[kk] * (tracer_dev[kk] + tracer_tmp);
          Real tracer_mean = tracer_dev[kk] *
                                 (1. - cldc_dev[kk] * f_act_conv_dev[kk]) -
                             cldc_dev[kk] * f_act_conv_dev[kk] * tracer_tmp;
          tracer_mean = mam4::max(0., tracer_mean);

          // Convective scavenging
          Real fracp_c =
              cmfdqr_dev[kk] * deltat /
              mam4::max(small_value_12,
                        cldc_dev[kk] * conicw_dev[kk] +
                            (cmfdqr_dev[kk] + dlf_dev[kk]) * deltat);
          fracp_c = mam4::utils::min_max_bound(0.0, 1.0, fracp_c) * cldc_dev[kk];
          Real srcc = 0;
          Real finc = 0;
          mam4::wetdep::wetdep_scavenging(
              2, is_strat_cloudborne, deltat, fracp_c, precabc, cldvcu_dev[kk],
              scavcoef_dev[kk], sol_factb, sol_factic_dev[kk], tracer_incu,
              tracer_mean, srcc, finc);

          // Stratiform scavenging
          Real fracp_s = precs_dev[kk] * deltat /
                         mam4::max(small_value_12,
                                   cwat_dev[kk] + precs_dev[kk] * deltat);
          fracp_s = mam4::utils::min_max_bound(0.0, 1.0, fracp_s);
          Real srcs = 0;
          Real fins = 0;
          mam4::wetdep::wetdep_scavenging(
              1, is_strat_cloudborne, deltat, fracp_s, precabs, cldvst_dev[kk],
              scavcoef_dev[kk], sol_factb, sol_facti, tracer_dev[kk],
              tracer_mean, srcs, fins);

          // Bound removal
          const Real rat = tracer_dev[kk] /
                           mam4::max(deltat * (srcc + srcs), small_value_36);
          if (rat < 1.0) {
            srcs *= rat;
            srcc *= rat;
          }
          const Real srct = (srcc + srcs) * omsm;

          // Diagnostics
          const Real fracp = deltat * srct /
                             mam4::max(cldvst_dev[kk] * tracer_dev[kk],
                                       small_value_36);
          fracis_dev[kk] = 1.0 - mam4::utils::min_max_bound(0.0, 1.0, fracp);
          iscavt_dev[kk] = -(srcc * finc + srcs * fins) * omsm;
          icscavt_dev[kk] = -(srcc * finc) * omsm;
          isscavt_dev[kk] = -(srcs * fins) * omsm;

          if (mam_prevap_resusp_optcc == 0) {
            bsscavt_dev[kk] = -(srcs * (1.0 - fins)) * omsm +
                              fracev_st * scavabs * gravit / pdel_dev[kk];
            // Update column fluxes for optcc == 0
            scavabs = scavabs * (1.0 - fracev_st) +
                      srcs * pdel_dev[kk] / gravit;
            scavabc = scavabc * (1.0 - fracev_cu) +
                      srcc * pdel_dev[kk] / gravit;
            precabs += (precs_dev[kk] - evaps_dev[kk]) * pdel_dev[kk] / gravit;
            precabc += (cmfdqr_dev[kk] - evapc_dev[kk]) * pdel_dev[kk] / gravit;
          } else {
            bsscavt_dev[kk] = -(srcs * (1.0 - fins)) * omsm;
            // Column fluxes for optcc >= 100
            const int next_k = mam4::min(kk + 1, nlev - 1);
            // Stratiform
            const Real arainx_s = mam4::max(cldvst_dev[next_k], small_value_2);
            Real resusp_s = mam4::wetdep::wetdep_resusp(
                1, mam_prevap_resusp_optcc, pdel_dev[kk], evaps_dev[kk],
                precabs, precabs_base, scavabs, precnums_base);
            const Real tmpa_s = mam4::max(0.0, evaps_dev[kk] * pdel_dev[kk] / gravit);
            Real precabx_tmp_s = mam4::utils::min_max_bound(0.0, precabs_base, precabs - tmpa_s);
            Real precabx_base_tmp_s = precabs_base;
            Real scavabx_tmp_s = scavabs;
            Real precnumx_base_tmp_s = precnums_base;
            if (precabx_tmp_s < 1.e-30) {
              precabx_tmp_s = 0.0;
              precabx_base_tmp_s = 0.0;
              if (mam_prevap_resusp_optcc <= 130) {
                scavabx_tmp_s = 0.0;
              }
            } else if (evaps_dev[kk] <= 0.0) {
              // no evap
            } else {
              if (mam_prevap_resusp_optcc <= 130) {
                scavabx_tmp_s = mam4::max(0.0, scavabs - resusp_s);
              }
            }
            wetdep_prevap_step(
                1, mam_prevap_resusp_optcc, pdel_dev[kk], precs_dev[kk],
                srcs, arainx_s, precabx_tmp_s, precabx_base_tmp_s,
                scavabx_tmp_s, precnumx_base_tmp_s, precabs, precabs_base,
                scavabs, precnums_base);

            // Convective
            const Real arainx_c = mam4::max(cldvcu_dev[next_k], small_value_2);
            Real resusp_c = mam4::wetdep::wetdep_resusp(
                2, mam_prevap_resusp_optcc, pdel_dev[kk], evapc_dev[kk],
                precabc, precabc_base, scavabc, precnumc_base);
            const Real tmpa_c = mam4::max(0.0, evapc_dev[kk] * pdel_dev[kk] / gravit);
            Real precabx_tmp_c = mam4::utils::min_max_bound(0.0, precabc_base, precabc - tmpa_c);
            Real precabx_base_tmp_c = precabc_base;
            Real scavabx_tmp_c = scavabc;
            Real precnumx_base_tmp_c = precnumc_base;
            if (precabx_tmp_c < 1.e-30) {
              precabx_tmp_c = 0.0;
              precabx_base_tmp_c = 0.0;
              if (mam_prevap_resusp_optcc <= 130) {
                scavabx_tmp_c = 0.0;
              }
            } else if (evapc_dev[kk] <= 0.0) {
              // no evap
            } else {
              if (mam_prevap_resusp_optcc <= 130) {
                scavabx_tmp_c = mam4::max(0.0, scavabc - resusp_c);
              }
            }
            wetdep_prevap_step(
                2, mam_prevap_resusp_optcc, pdel_dev[kk], cmfdqr_dev[kk],
                srcc, arainx_c, precabx_tmp_c, precabx_base_tmp_c,
                scavabx_tmp_c, precnumx_base_tmp_c, precabc, precabc_base,
                scavabc, precnumc_base);
          }
        }
      });

  auto copy_to_host = [&](std::string name, mam4::ColumnView dev) {
    std::vector<Real> vec(nlev);
    auto host = Kokkos::create_mirror_view(dev);
    Kokkos::deep_copy(host, dev);
    for (int i = 0; i < nlev; ++i)
      vec[i] = host[i];
    output.set(name, vec);
  };
  // Create mirror views for output arrays
  copy_to_host("fracis", fracis_dev);
  copy_to_host("scavt", scavt_dev);
  copy_to_host("iscavt", iscavt_dev);
  copy_to_host("icscavt", icscavt_dev);
  copy_to_host("isscavt", isscavt_dev);
  copy_to_host("bcscavt", bcscavt_dev);
  copy_to_host("bsscavt", bsscavt_dev);
  copy_to_host("rcscavt", rcscavt_dev);
  copy_to_host("rsscavt", rsscavt_dev);
}

void test_wetdepa_v2(std::unique_ptr<Ensemble> &ensemble) {
  ensemble->process([&](const Input &input, Output &output) {
    test_wetdepa_v2_process(input, output);
  });
}
