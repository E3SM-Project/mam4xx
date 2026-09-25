#ifndef MAMBOXX_MICROPHYS_HPP
#define MAMBOXX_MICROPHYS_HPP

#include "config.hpp"

namespace mamboxx {

class Microphys {
  Microphys() = default;
  ~Microphys() = default;

  Microphys(const Microphys&) = delete;
  Microphys& operator=(const Microphys&) = delete;

  void initialize(const MicrophysConfig &config);
  void step(Real dt);
  void finalize();

private:
  MicrophysConfig config;
};

} // namespace mamboxx

#endif
