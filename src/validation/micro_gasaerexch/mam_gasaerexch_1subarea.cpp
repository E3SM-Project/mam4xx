// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::gasaerexch;

void mam_gasaerexch_1subarea(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Extract scalar values from input arrays
    const int jtsubstep = input.get_array("jtsubstep")[0];
    const Real dtsubstep = input.get_array("dtsubstep")[0];
    const Real temp = input.get_array("temp")[0];
    const Real pmid = input.get_array("pmid")[0];
    const Real aircon = input.get_array("aircon")[0];
    const int n_mode = input.get_array("n_mode")[0];

    // Extract array values from input arrays
    auto qgas_cur_vec = input.get_array("qgas_cur");
    auto qgas_avg_vec = input.get_array("qgas_avg");
    auto qgas_netprod_otrproc_vec = input.get_array("qgas_netprod_otrproc");
    auto qaer_cur_vec = input.get_array("qaer_cur");
    auto qnum_cur_vec = input.get_array("qnum_cur");
    auto qwtr_cur_vec = input.get_array("qwtr_cur");
    auto dgn_awet_vec = input.get_array("dgn_awet");
    auto uptkaer_vec = input.get_array("uptkaer");

    Real qgas_cur[max_gas];
    Real qgas_avg[max_gas];
    Real qgas_netprod_otrproc[max_gas];
    Real qaer_cur[max_aer][max_mode];
    Real qnum_cur[max_mode];
    Real qwtr_cur[max_mode];
    Real dgn_awet[max_mode];
    Real uptkaer[max_gas][max_mode];
    Real uptkrate_h2so4 = 0;

    // Copy input vectors to local arrays
    for (int i = 0; i < max_gas; ++i) {
      qgas_cur[i] = qgas_cur_vec[i];
      qgas_avg[i] = qgas_avg_vec[i];
      qgas_netprod_otrproc[i] = qgas_netprod_otrproc_vec[i];
    }

    for (int i = 0; i < max_mode; ++i) {
      qnum_cur[i] = qnum_cur_vec[i];
      qwtr_cur[i] = qwtr_cur_vec[i];
      dgn_awet[i] = dgn_awet_vec[i];
    }

    for (int j = 0; j < max_mode; ++j) {
      for (int i = 0; i < max_aer; ++i) {
        qaer_cur[i][j] = qaer_cur_vec[j * max_aer + i];
      }
    }

    for (int j = 0; j < max_mode; ++j) {
      for (int i = 0; i < max_gas; ++i) {
        uptkaer[i][j] = uptkaer_vec[j * max_gas + i];
      }
    }

    // Call the function
    mam4::mam_gasaerexch_1subarea(jtsubstep, dtsubstep, temp, pmid, aircon,
                                  n_mode, qgas_cur, qgas_avg,
                                  qgas_netprod_otrproc, qaer_cur, qnum_cur,
                                  dgn_awet, uptkaer, uptkrate_h2so4);

    // Prepare output arrays
    std::vector<Real> qgas_cur_out(max_gas);
    std::vector<Real> qgas_avg_out(max_gas);
    std::vector<Real> qaer_cur_out(max_aer * max_mode);
    std::vector<Real> qnum_cur_out(max_mode);
    std::vector<Real> qwtr_cur_out(max_mode);
    std::vector<Real> dgn_a_out(max_mode, 0);
    std::vector<Real> dgn_awet_out(max_mode);
    std::vector<Real> wetdens_out(max_mode, 0);
    std::vector<Real> uptkaer_out(max_gas * max_mode);

    for (int i = 0; i < max_gas; ++i) {
      qgas_cur_out[i] = qgas_cur[i];
      qgas_avg_out[i] = qgas_avg[i];
    }

    for (int i = 0; i < max_mode; ++i) {
      qnum_cur_out[i] = qnum_cur[i];
      qwtr_cur_out[i] = qwtr_cur[i];
      dgn_awet_out[i] = dgn_awet[i];
    }

    for (int j = 0; j < max_mode; ++j) {
      for (int i = 0; i < max_aer; ++i) {
        qaer_cur_out[j * max_aer + i] = qaer_cur[i][j];
      }
    }

    for (int j = 0; j < max_mode; ++j) {
      for (int i = 0; i < max_gas; ++i) {
        uptkaer_out[j * max_gas + i] = uptkaer[i][j];
      }
    }

    // Set output values
    output.set("qgas_cur", qgas_cur_out);
    output.set("qgas_avg", qgas_avg_out);
    output.set("qaer_cur", qaer_cur_out);
    output.set("qnum_cur", qnum_cur_out);
    output.set("qwtr_cur", qwtr_cur_out);
    output.set("dgn_a", dgn_a_out);
    output.set("dgn_awet", dgn_awet_out);
    output.set("wetdens", wetdens_out);
    output.set("uptkaer", uptkaer_out);
    output.set("uptkrate_h2so4", uptkrate_h2so4);
  });
}
