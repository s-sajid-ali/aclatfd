#include "io.hpp"

/* --------------------------------------------------------------------- */
/*!
  finalize by destroying data structures

  \param   lctx - local context
  \param   sctx - subcomm context
  \param   gctx - global context
  \return  ierr - PetscErrorCode

*/
PetscErrorCode save_state(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx, std::string& name){

  PetscViewer    hdf5_viewer;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, const_cast<char *>(name.c_str()), FILE_MODE_WRITE, &hdf5_viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(hdf5_viewer, "global_members");CHKERRQ(ierr);
  ierr = VecView(gctx.phi_global_local, hdf5_viewer);CHKERRQ(ierr);
  ierr = VecView(gctx.rho_global_local, hdf5_viewer);CHKERRQ(ierr);
  for (PetscInt i=0; i<gctx.nsubcomms; i++) {
    ierr = VecView(gctx.phi_global_subcomm[i], hdf5_viewer);CHKERRQ(ierr);
    ierr = VecView(gctx.rho_global_subcomm[i], hdf5_viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(hdf5_viewer);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
};
