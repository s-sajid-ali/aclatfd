#include "utils.hpp"
#include "comms.hpp"

#include <algorithm>

/* --------------------------------------------------------------------- */
/*!
  Initialize solver subcomms

  \param   sctx - subcomm context
  \param   gctx - global context
  \return  ierr - PetscErrorCode

*/
PetscErrorCode init_solversubcomms(SubcommCtx& sctx, GlobalCtx& gctx) {

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = PetscSubcommCreate(PETSC_COMM_WORLD, &(sctx.solverpsubcomm));CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(sctx.solverpsubcomm, gctx.nsubcomms);CHKERRQ(ierr);
  ierr = PetscSubcommSetType(sctx.solverpsubcomm, PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);

  ierr = PetscSubcommGetChild(sctx.solverpsubcomm, &(sctx.solversubcomm));CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&(gctx.global_rank),
      &(sctx.solversubcommid),
      1,
      MPIU_INT,
      MPI_MAX,
      sctx.solversubcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(sctx.solversubcomm, &(sctx.solver_rank));CHKERRQ(ierr);
  ierr = MPI_Comm_size(sctx.solversubcomm, &(sctx.solver_size));CHKERRQ(ierr);

  /* collect the subcommids on each MPI rank & remove duplicates */
  gctx.sids.resize(gctx.global_size);
  ierr = MPI_Allgather(&sctx.solversubcommid,
      1,
      MPI_INT,
      gctx.sids.data(),
      1,
      MPI_INT,
      PETSC_COMM_WORLD);CHKERRMPI(ierr);
  std::sort(gctx.sids.begin(), gctx.sids.end());
  gctx.sids.erase( std::unique(gctx.sids.begin(), gctx.sids.end()), gctx.sids.end() );

  if (gctx.debug) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nsolver-subcomms have been created!\n");CHKERRQ(ierr);
    ierr = PetscSubcommView(sctx.solverpsubcomm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
        "size of gct.sids vector on global-rank %d is : %d\n",
        gctx.global_rank,
        gctx.sids.size());CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}


/* --------------------------------------------------------------------- */
/*!
  Initialize sequential vectors on each MPI rank

  \param   lctx - local context
  \param   gctx - global context
  \return  ierr - PetscErrorCode

*/

PetscErrorCode init_localvecs(LocalCtx& lctx, GlobalCtx& gctx){

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = VecCreate(PETSC_COMM_SELF, &lctx.seqphi);CHKERRQ(ierr);
  ierr = VecSetType(lctx.seqphi, gctx.vectype.c_str());CHKERRQ(ierr);
  ierr = VecSetSizes(lctx.seqphi, PETSC_DECIDE, gctx.nsize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(lctx.seqphi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(lctx.seqphi), "seqphi_on_lctx");CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF, &lctx.seqrho);CHKERRQ(ierr);
  ierr = VecSetType(lctx.seqrho, gctx.vectype.c_str());CHKERRQ(ierr);
  ierr = VecSetSizes(lctx.seqrho, PETSC_DECIDE, gctx.nsize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(lctx.seqrho);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(lctx.seqrho), "seqrho_on_lctx");CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


/* --------------------------------------------------------------------- */
/*!
  Initialize vectors on each solver-subcommunicator

  \param   sctx - subcomm context
  \param   gctx - global context
  \return  ierr - PetscErrorCode
  */
PetscErrorCode init_subcommvecs(SubcommCtx& sctx, GlobalCtx& gctx){
  PetscInt       size;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCreate(sctx.solversubcomm, &sctx.phi_subcomm);CHKERRQ(ierr);
  ierr = VecSetType(sctx.phi_subcomm, gctx.vectype.c_str());CHKERRQ(ierr);
  ierr = VecSetSizes(sctx.phi_subcomm, PETSC_DECIDE, gctx.nsize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(sctx.phi_subcomm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(sctx.phi_subcomm), "phi_subcomm_on_sctx");CHKERRQ(ierr);

  ierr = VecCreate(sctx.solversubcomm, &sctx.rho_subcomm);CHKERRQ(ierr);
  ierr = VecSetType(sctx.rho_subcomm, gctx.vectype.c_str());CHKERRQ(ierr);
  ierr = VecSetSizes(sctx.rho_subcomm, PETSC_DECIDE, gctx.nsize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(sctx.rho_subcomm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(sctx.rho_subcomm), "rho_subcomm_on_sctx");CHKERRQ(ierr);

  if (gctx.debug) {
    ierr = VecGetSize(sctx.rho_subcomm, &size);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)sctx.rho_subcomm),
        "Hi there from solver-subcomm with id %d, the size of subcomm vec here is %d\n",
        sctx.solversubcommid,
        size);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}


/* --------------------------------------------------------------------- */
/*!
  Initialize global (alias of local) to subcomm scatters

  \param   sctx - subcomm context
  \param   gctx - global context
  \return  ierr - PetscErrorCode

*/
PetscErrorCode init_global_subcomm_scatters(SubcommCtx& sctx, GlobalCtx& gctx){

  PetscInt       start;
  PetscInt       localsize;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  /* localsize should be gctx->nsize, but adding a check doesn't hurt */
  ierr = VecGetLocalSize(gctx.rho_global_local, &localsize);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(gctx.rho_global_local, &start,NULL);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF, localsize, start, 1, &gctx.ix_scat_glocal_to_subcomms);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, localsize,     0, 1, &gctx.iy_scat_glocal_to_subcomms);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)(gctx.ix_scat_glocal_to_subcomms), "ix_scat_glocal_to_subcomms");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(gctx.iy_scat_glocal_to_subcomms), "iy_scat_glocal_to_subcomms");CHKERRQ(ierr);

  /* resize to ensure we only create as many scatters
     as the number of subcomms ! */
  gctx.scat_glocal_to_subcomms.resize(gctx.nsubcomms);

  for (PetscInt i=0; i<gctx.nsubcomms; i++){
    ierr = VecScatterCreate(gctx.rho_global_local,
        gctx.ix_scat_glocal_to_subcomms,
        gctx.rho_global_subcomm[i],
        gctx.iy_scat_glocal_to_subcomms,
        &(gctx.scat_glocal_to_subcomms[i]));CHKERRQ(ierr);
  }

  if (gctx.debug) {
    for (PetscInt i=0; i<gctx.nsubcomms; i++){
      ierr = PetscPrintf(PETSC_COMM_WORLD, "global-to-subcomm scatter\n");CHKERRQ(ierr);
      ierr = VecScatterView(gctx.scat_glocal_to_subcomms[i], PETSC_VIEWER_STDOUT_WORLD);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}


/* --------------------------------------------------------------------- */
/*!
  Initialize subcomm (alias of local) to local scatters

  \param   lctx - local context
  \param   sctx - subcomm context
  \param   gctx - global context
  \return  ierr - PetscErrorCode

*/
PetscErrorCode init_subcomm_local_scatters(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx){

  PetscInt       start;
  PetscInt       localsize;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  /* localsize should be gctx->nsize, but adding a check doesn't hurt */
  ierr = VecGetLocalSize(sctx.phi_subcomm_local, &localsize);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(sctx.phi_subcomm_local, &start,NULL);CHKERRQ(ierr);

  ierr = ISCreateStride(PETSC_COMM_SELF, localsize, start, 1, &sctx.ix_scat_subcomms_to_local);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, localsize,     0, 1, &sctx.iy_scat_subcomms_to_local);CHKERRQ(ierr);

  /* This scatter will always be run as a SCATTER_REVERSE,
     perhaps the naming terminilogy may be updated to prevent
     any confusion in the future.*/
  ierr = VecScatterCreate(sctx.phi_subcomm_local,
      sctx.ix_scat_subcomms_to_local,
      sctx.phi_subcomm,
      sctx.iy_scat_subcomms_to_local,
      &sctx.scat_subcomm_to_local);CHKERRQ(ierr);

  if (gctx.debug) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "subcomm-to-local scatter\n");CHKERRQ(ierr);
    ierr = VecScatterView(sctx.scat_subcomm_to_local, PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)sctx.phi_subcomm)));
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    ierr = PetscBarrier(NULL);
  }

  PetscFunctionReturn(ierr);
}


/* --------------------------------------------------------------------- */
/*!
  finalize by destroying data structures

  \param   lctx - local context
  \param   sctx - subcomm context
  \param   gctx - global context
  \return  ierr - PetscErrorCode

*/
PetscErrorCode finalize(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx){

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  /* Destroy global aliases of local vectors */
  ierr = VecDestroy(&(gctx.phi_global_local));CHKERRQ(ierr);
  ierr = VecDestroy(&(gctx.rho_global_local));CHKERRQ(ierr);

  /* Destroy global aliases of subcomm vectors */
  for (PetscInt i=0; i<gctx.nsubcomms; i++){
    ierr = VecDestroy(&(gctx.phi_global_subcomm[i]));CHKERRQ(ierr);
    ierr = VecDestroy(&(gctx.rho_global_subcomm[i]));CHKERRQ(ierr);
  }

  /* Destroy subcomm vectors */
  ierr = VecDestroy(&(sctx.phi_subcomm));CHKERRQ(ierr);
  ierr = VecDestroy(&(sctx.rho_subcomm));CHKERRQ(ierr);

  /* Destroy global (alias of local) to subcomm scatters */
  for (PetscInt i=0; i<gctx.nsubcomms; i++){
    ierr = VecScatterDestroy(&(gctx.scat_glocal_to_subcomms[i]));CHKERRQ(ierr);
  }
  ierr = ISDestroy(&gctx.ix_scat_glocal_to_subcomms);CHKERRQ(ierr);
  ierr = ISDestroy(&gctx.iy_scat_glocal_to_subcomms);CHKERRQ(ierr);

  /* Destroy subcomm aliases of local vectors */
  ierr = VecDestroy(&sctx.phi_subcomm_local);CHKERRQ(ierr);
  ierr = VecDestroy(&sctx.rho_subcomm_local);CHKERRQ(ierr);

  /* Destroy subcomm vectors */
  ierr = VecDestroy(&(lctx.seqphi));CHKERRQ(ierr);
  ierr = VecDestroy(&(lctx.seqrho));CHKERRQ(ierr);

  /* Destroy subcomm (alias of local) to local scatters */
  ierr = VecScatterDestroy(&sctx.scat_subcomm_to_local);CHKERRQ(ierr);
  ierr = ISDestroy(&sctx.ix_scat_subcomms_to_local);CHKERRQ(ierr);
  ierr = ISDestroy(&sctx.iy_scat_subcomms_to_local);CHKERRQ(ierr);

  /* Destroy subcomms */
  ierr = PetscSubcommDestroy(&(sctx.solverpsubcomm));CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}


