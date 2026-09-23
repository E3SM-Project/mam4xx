// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <string>

#ifndef MAMBOXX_HPP
#define MAMBOXX_HPP

namespace mamboxx {

struct Config {
};

// Reads a mamboxx config file, returning a Config struct storing its contents.
Config read_config(const std::string &filename);

}

#endif
