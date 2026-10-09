#include "mamboxx.hpp"
#include <mam4xx/mam4_types.hpp>

#include <ekat_assert.hpp>

namespace mamboxx {

namespace {

// Column allocation machinery -- copied from testing.cpp

// A simple memory allocation pool for standalone ColumnViews to be used in
// (e.g.) unit tests. A ColumnPool manages a number of ColumnViews with a fixed
// number of vertical levels.
class ColumnPool {
  size_t num_levels_;          // number of vertical levels per column (fixed)
  size_t num_cols_;            // number of allocated columns
  std::vector<int> col_used_;  // columns that are being used already
  std::vector<Real *> memory_; // per-column memory itself (allocated on device)
public:
  // constructs a column pool with the given initial number of columns, each
  // with the given number of vertical levels.`
  ColumnPool(size_t num_vertical_levels, size_t initial_num_columns = 64)
      : num_levels_(num_vertical_levels), num_cols_(initial_num_columns),
        col_used_(initial_num_columns, 0),
        memory_(initial_num_columns, nullptr) {
    for (size_t i = 0; i < num_cols_; ++i) {
      memory_[i] = reinterpret_cast<Real *>(Kokkos::kokkos_malloc(
          "Column pool", sizeof(Real) * num_vertical_levels));
    }
  }

  // destructor
  ~ColumnPool() {
    for (size_t i = 0; i < num_cols_; ++i) {
      Kokkos::kokkos_free(memory_[i]);
    }
  }

  // returns a "fresh" (unused) ColumnView from the ColumnPool, marking it as
  // used, and allocating additional memory if needed)
  mam4::ColumnView column_view() {
    // find the first unused column
    size_t i;
    for (i = 0; i < num_cols_; ++i) {
      if (!col_used_[i])
        break;
    }
    if (i == num_cols_) { // all columns in the pool are in use!
      // double the number of allocated columns in the pool
      size_t new_num_cols = 2 * num_cols_;
      col_used_.resize(new_num_cols, 0);
      memory_.resize(new_num_cols, nullptr);
      for (size_t i = num_cols_; i < new_num_cols; ++i) {
        memory_[i] = reinterpret_cast<Real *>(
            Kokkos::kokkos_malloc(sizeof(Real) * num_levels_));
      }
      num_cols_ = new_num_cols;
    }

    col_used_[i] = 1;
    auto column = mam4::ColumnView(memory_[i], num_levels_);
    Kokkos::deep_copy(column, 0);
    return column;
  }
};

// column pools, organized by column resolution
std::map<size_t, std::unique_ptr<ColumnPool>> pools_{};

void destroy_pools() {
  pools_.clear();
}

mam4::ColumnView create_column_view(int num_levels) {
  if (pools_.empty()) {
    std::atexit(destroy_pools);
  }

  // find a column pool for the given number of vertical levels
  auto iter = pools_.find(num_levels);
  if (iter == pools_.end()) {
    auto result = pools_.emplace(
        num_levels, std::unique_ptr<ColumnPool>(new ColumnPool(num_levels)));
    iter = result.first;
  }
  return iter->second->column_view();
}

struct State {
  AtmosphereConfig atmosphere;
  AerosolState     aerosols;
  GasState         gases;
};

void initialize(const Config &config, State &state) {
  size_t n_lev = config.atmosphere.temperature.size();
  EKAT_REQUIRE(n_lev <= mam4::nlev,
      "requested number of vertical levels (" << n_lev <<
      ") exceeds compile-time limit (" << mam4::nlev <<
      ") -- reconfigure with MAM4XX_NUM_VERTICALS set to a higher limit");
}

void step(State &state, Real dt) {
  size_t n_lev = state.atmosphere.temperature.size();
  size_t k_top = 0, k_bottom = n_lev; // FIXME: provide specific k for box model?
 
  // For guidance here, see MAMMicrophysics::run_impl in EAMxx.

  // TODO: add chlorine loading for linoz
  // TODO: add elevated emissions
  // TODO: add insolation based on orbital parameters/date and location (or just directly)
 
  // MAMMicrophysics::run_microphysics_kernels

  // TODO: external forcіng
  //mam4::ColumnView external_forcing = create_column_view(n_lev);
  // TODO: set invariants (?)
  // TODO: compute o3_col_dens
  // TODO: do photolysis table lookups
  // TODO: sethet
  // TODO: drydep_xactive
 

}

} // anonymous namespace

Result run(const Config &config) {
  Result result = {};

  State state = {};
  initialize(config, state);

  for (int i = 0; i < config.time.nstep; ++i) {
    step(state, config.time.dt);
  }

  return result;
}

}
