#ifndef MAMBOXX_PYTHON_HPP
#define MAMBOXX_PYTHON_HPP

#include "mamboxx.hpp"

namespace mamboxx {

Config read_config(const std::string &filename);
bool   run_verification_checks(const Config &config, const State &state);

}

#endif
