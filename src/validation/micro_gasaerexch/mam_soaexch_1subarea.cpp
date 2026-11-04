// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <mam4xx/mam4.hpp>
#include <validation.hpp>

using namespace skywalker;
using namespace mam4::gasaerexch;

void mam_soaexch_1subarea(Ensemble *ensemble) {
  ensemble->process([=](const Input &input, Output &output) {
    // Extract scalar values from input arrays
    const Real dtsubstep = input.get_array("dtsubstep")[0];
    const Real temp = input.get_array("temp")[0];
    const Real pmid = input.get_array("pmid")[0];

    // Extract array values from input arrays
    auto qgas_cur_vec = input.get_array("qgas_cur");
    auto qgas_avg_vec = input.get_array("qgas_avg");
    auto qaer_cur_vec = input.get_array("qaer_cur");
    auto qnum_cur_vec = input.get_array("qnum_cur");
    auto qwtr_cur_vec = input.get_array("qwtr_cur");
    auto uptkaer_vec = input.get_array("uptkaer");

    Real qaer_cur[max_aer][max_mode];
    Real uptkaer[max_gas][max_mode];

    for (int idx = 0; idx < max_aer; ++idx) {
      for (int mode = 0; mode < max_mode; ++mode) {
        qaer_cur[idx][mode] = qaer_cur_vec[idx * max_mode + mode];
      }
    }
    for (int idx = 0; idx < max_gas; ++idx) {
      for (int mode = 0; mode < max_mode; ++mode) {
        uptkaer[idx][mode] = uptkaer_vec[idx * max_mode + mode];
      }
    }

    // Call the function
    mam_soaexch_1subarea(dtsubstep, temp, pmid, qgas_cur_vec.data(),
                         qgas_avg_vec.data(), qaer_cur, uptkaer);

    for (int idx = 0; idx < max_aer; ++idx) {
      for (int mode = 0; mode < max_mode; ++mode) {
        qaer_cur_vec[idx * max_mode + mode] = qaer_cur[idx][mode];
      }
    }

    output.set("qgas_cur", qgas_cur_vec);
    output.set("qgas_avg", qgas_avg_vec);
    output.set("qwtr_cur", qwtr_cur_vec);
    output.set("qnum_cur", qnum_cur_vec);
    output.set("qaer_cur", qaer_cur_vec);
  });
}
