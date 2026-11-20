#include "aerosol_cloud_interaction.hpp"

namespace mam4::aerosol_cloud_interaction {

DropMixNuc::DropMixNuc(const DropMixNucConfig &config):
  config_(config) {
}

DropMixNuc::DropMixNuc(const DropMixNucConfig &config, WorkArrays &work_arrays) {
}

DropMixNuc::~DropMixNuc() {
}

void DropMixNuc::step(const mam4::Atmosphere &atmosphere, mam4::Prognostics &prognostics, Real dt) {
}

void DropMixNuc::set_work_arrays(const WorkArrays &work_arrays, bool assume_ownership) {
}

} // namespace mam4::aerosol_cloud_interaction
