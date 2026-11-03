#ifndef MAM4XX_MO_SETEXT_HPP
#define MAM4XX_MO_SETEXT_HPP

#include "mam4_types.hpp"

namespace mam4 {

namespace mo_setext {
using View1D = DeviceType::view_1d<Real>;
using View2D = DeviceType::view_2d<Real>;
using View3D = DeviceType::view_3d<Real>;
constexpr int extfrc_cnt = 9;
constexpr int extcnt = 9; //, & ! number of species with external forcing
// MAX_NUM_SECTIONS: Maximum number of sections in forcing data. Increase this
// number if needed.
constexpr int MAX_NUM_SECTIONS = 4;
struct Forcing {
  // This index is in Fortran format. i.e. starts in 1
  int frc_ndx;
  bool file_alt_data;
  View1D fields_data[MAX_NUM_SECTIONS];
  int nsectors;
};

KOKKOS_INLINE_FUNCTION
void extfrc_set(const ThreadTeam &team, const Forcing *forcings,
                const View2D &frcing) {
  const int pver = mam4::nlev;

  /*--------------------------------------------------------
   ... form the external forcing
  --------------------------------------------------------*/
  // param[in] forcings(extcnt) array with a list of Forcing object.
  // @param[out] frcing(ncol,pver,extcnt)   insitu forcings [molec/cm^3/s]
  // Note: we do not need zint to compute frcing
  // const ColumnView &zint

  constexpr Real zero = 0.0;

  // frcing(:,:,:) = zero;
  if (extfrc_cnt < 1 || extcnt < 1) {
    return;
  }

  /*--------------------------------------------------------
  ! ... set non-zero forcings
  !--------------------------------------------------------*/
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, pver), [&](int kk) {
    for (int mm = 0; mm < extfrc_cnt; ++mm) {
      // Fortran to C++ indexing
      auto forcing_mm = forcings[mm];
      const int nn = forcing_mm.frc_ndx - 1;
      frcing(nn, kk) = zero;

      for (int isec = 0; isec < forcing_mm.nsectors; ++isec) {
        if (forcing_mm.file_alt_data) {
          frcing(nn, kk) += forcing_mm.fields_data[isec](pver - 1 - kk);
        } else {
          // forcings(mm)%fields(isec)%data(:ncol,:,lchnk)
          frcing(nn, kk) += forcing_mm.fields_data[isec](kk);
        }
      } // isec
    }   // end mm
  });
} // extfrc_set

KOKKOS_INLINE_FUNCTION
void setext(const ThreadTeam &team, const Forcing *forcings,
            const View2D &extfrc) // ! out
{
  // Note: we do not need setext.
  // Because I removed diagnostic variables(i.e., output variable in outfld),
  // setext and extfrc_set are equivalent.
  /*--------------------------------------------------------
       ... for this latitude slice:
           - form the production from datasets
           - form the nox (xnox) production from lighing
           - form the nox (xnox) production from airplanes
           - form the co production from airplanes
  --------------------------------------------------------*/

  // param[in] forcings(extcnt) array with a list of Forcing object.
  // @param[out]  extfrc(ncol,pver,extcnt)    ! the "extraneous" forcing

  /*--------------------------------------------------------
    !     ... local variables
    !--------------------------------------------------------
    ! variables for output. in current MAM4 they are not calculated and are
    assigned zero real(r8), dimension(ncol,pver) :: no_lgt, no_air, co_air */

  /*--------------------------------------------------------
  !     ... set frcing from datasets
  !--------------------------------------------------------*/
  extfrc_set(team, forcings,
             extfrc); // out

  /*--------------------------------------------------------
       ... set nox production from lighting
           note: from ground to cloud top production is c shaped

   FORTRAN refactor: nox is not included in current MAM4
   the related code are removed but outfld is kept for BFB testing
  --------------------------------------------------------*/
  // no_lgt(:,:) = 0._r8
  // call outfld( 'NO_Lightning', no_lgt(:ncol,:), ncol, lchnk )

  // ! FORTRAN refactor: in the subroutine airpl_set, has_airpl_src is false,
  // ! the subroutine only has two outfld calls that output zero
  // ! remove the subroutine call and move out the zero outfld
  // no_air(:,:) = 0._r8
  // co_air(:,:) = 0._r8
  // call outfld( 'NO_Aircraft',  no_air(:ncol,:), ncol, lchnk )
  // call outfld( 'CO_Aircraft',  co_air(:ncol,:), ncol, lchnk )

} // setext

} // namespace mo_setext

} // end namespace mam4

#endif
