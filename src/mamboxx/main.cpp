// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include "mamboxx.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

struct CommandLineArgs {
  bool valid;
  std::string error;

  std::string filename;
};

const char *usage = "mamboxx: a box model for mam4xx's aerosol processes\n\
mamboxx: usage:\n\
mamboxx [options] <config-file>\n";

CommandLineArgs parse_args(int argc, char *argv[]) {
  CommandLineArgs args = {};
  if (argc < 2) { // no args!
    args.error = usage;
  }

  args.filename = argv[1];

  args.valid = true;
  return args;
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  auto args = parse_args(argc, argv);
  if (!args.valid) {
    fprintf(stderr, "%s", args.error.c_str());
    exit(0);
  }

  auto config = mamboxx::read_config(args.filename);

  return 0;
}
