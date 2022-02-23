#include "alias.hpp"
#include "comms.hpp"

#include <ios>

/* --------------------------------------------------------------------- */
/*!
  Initialize global aliases of local vectors
  \param   lctx - local context
  \param   gctx - global context
  \return  void - nothing
  */
PetscErrorCode determine_veccreatewitharray_func(GlobalCtx& gctx) {
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;

  if (gctx.VecCreate_type_WithArray==nullptr) {

    if (gctx.vectype=="standard") {
      gctx.VecCreate_type_WithArray = VecCreateMPIWithArray;
    } else if (gctx.vectype=="cuda") {
#if defined(PETSC_HAVE_CUDA)
      gctx.VecCreate_type_WithArray = VecCreateMPICUDAWithArray;
#endif
    } else if (gctx.vectype=="kokkos") {
#if defined(PETSC_HAVE_KOKKOS)
      gctx.VecCreate_type_WithArray = VecCreateMPIKokkosWithArray;
#endif
    } else {
      SETERRQ(PETSC_COMM_WORLD,
          PETSC_ERR_SUP,
          "The requested vector type is not supported yet (HIP/SYCL/etc)");
    }
  }

  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------- */
/*!
  Initialize global aliases of local vectors
  \param   lctx - local context
  \param   gctx - global context
  \return  ierr - PetscErrorCode
  */
PetscErrorCode init_global_local_aliases(LocalCtx& lctx, GlobalCtx& gctx){

  PetscInt           phi_localsize;   /* size of vector on this MPI rank */
  PetscInt           rho_localsize;   /* size of vector on this MPI rank */
  PetscScalar const  *d_phi_val;      /* read-only pointer to vector's contents on device */
  PetscScalar const  *d_rho_val;      /* read-only pointer to vector's contents on device */
  PetscMemType       mtype_phi;
  PetscMemType       mtype_rho;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;

  /* Get size of local vector */
  ierr = VecGetLocalSize(lctx.seqphi, &phi_localsize);CHKERRQ(ierr);
  ierr = VecGetLocalSize(lctx.seqrho, &rho_localsize);CHKERRQ(ierr);

  /* Get access to local vector */
  ierr = VecGetArrayReadAndMemType(lctx.seqphi, &d_phi_val, &mtype_phi);CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(lctx.seqrho, &d_rho_val, &mtype_rho);CHKERRQ(ierr);

  /* consistency check */
  if (gctx.debug) {
    if (mtype_phi != mtype_rho) SETERRQ(PETSC_COMM_WORLD,
        PETSC_ERR_ARG_NOTSAMETYPE,
        "phi and rho vectors don't have the memory type!");
    if (!gctx.VecCreate_type_WithArray) SETERRQ(PETSC_COMM_WORLD,
        PETSC_ERR_ARG_BADPTR,
        "the function pointer in gctx is invalid!");
  }

  ierr = gctx.VecCreate_type_WithArray(PETSC_COMM_WORLD,
      1,
      phi_localsize,
      PETSC_DECIDE,
      d_phi_val,
      &gctx.phi_global_local);CHKERRQ(ierr);
  ierr = gctx.VecCreate_type_WithArray(PETSC_COMM_WORLD,
      1,
      rho_localsize,
      PETSC_DECIDE,
      d_rho_val,
      &gctx.rho_global_local);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)(gctx.phi_global_local), "phi_global_local_on_gctx");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(gctx.rho_global_local), "rho_global_local_on_gctx");CHKERRQ(ierr);

  /* Restore local vector arrays */
  ierr = VecRestoreArrayReadAndMemType(lctx.seqphi, &d_phi_val);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(lctx.seqrho, &d_rho_val);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* --------------------------------------------------------------------- */
/*!
  Initialize global aliases of subcomm vectors
  \param   sctx - subcomm context
  \param   gctx - global context
  \return  ierr - PetscErrorCode
  */
PetscErrorCode init_global_subcomm_aliases(SubcommCtx& sctx, GlobalCtx& gctx){

  PetscInt           phi_localsize;   /* size of vector on this MPI rank */
  PetscInt           rho_localsize;   /* size of vector on this MPI rank */
  PetscScalar const  *d_phi_val;      /* read-only pointer to vector's contents on device */
  PetscScalar const  *d_rho_val;      /* read-only pointer to vector's contents on device */
  PetscMemType       mtype_phi;
  PetscMemType       mtype_rho;
  PetscInt           size;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;

  /* Get size of local vector */
  ierr = VecGetLocalSize(sctx.phi_subcomm, &phi_localsize);CHKERRQ(ierr);
  ierr = VecGetLocalSize(sctx.rho_subcomm, &rho_localsize);CHKERRQ(ierr);

  /* Get access to local vector */
  ierr = VecGetArrayReadAndMemType(sctx.phi_subcomm, &d_phi_val, &mtype_phi);CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(sctx.rho_subcomm, &d_rho_val, &mtype_rho);CHKERRQ(ierr);

  /* resize to ensure we only create as many aliases
     as the number of subcomms ! */
  gctx.phi_global_subcomm.resize(gctx.nsubcomms);
  gctx.rho_global_subcomm.resize(gctx.nsubcomms);

  ierr = PetscBarrier(NULL);CHKERRQ(ierr);
  for (PetscInt i=0; i<gctx.nsubcomms; i++){

    if (gctx.sids[i]==sctx.solversubcommid) {

      if (gctx.debug){
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PetscObjectComm((PetscObject)sctx.phi_subcomm),
            "Hi there from solver-subcomm number %d, with subcommid of %d\n",
            i,
            gctx.sids[i]);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
        if (mtype_phi != mtype_rho) SETERRQ(PETSC_COMM_WORLD,
            PETSC_ERR_ARG_NOTSAMETYPE,
            "phi and rho vectors don't have the memory type!");
        if (!gctx.VecCreate_type_WithArray) SETERRQ(PETSC_COMM_WORLD,
            PETSC_ERR_ARG_BADPTR,
            "the function pointer in gctx is invalid!");
      }

      /* Create global aliases of subcomm vectors */
      ierr = gctx.VecCreate_type_WithArray(PETSC_COMM_WORLD,
          1,
          phi_localsize,
          PETSC_DECIDE,
          d_phi_val,
          &(gctx.phi_global_subcomm[i]));CHKERRQ(ierr);
      ierr = gctx.VecCreate_type_WithArray(PETSC_COMM_WORLD,
          1,
          rho_localsize,
          PETSC_DECIDE,
          d_rho_val,
          &(gctx.rho_global_subcomm[i]));CHKERRQ(ierr);

    } else {

      /* ranks outside the subcomm whose vector is being aliased
         do not contribute any values to the global alias,
         yet they must participate in collective calls */
      ierr = gctx.VecCreate_type_WithArray(PETSC_COMM_WORLD,
          1,
          0,
          PETSC_DECIDE,
          NULL,
          &(gctx.phi_global_subcomm[i]));CHKERRQ(ierr);
      ierr = gctx.VecCreate_type_WithArray(PETSC_COMM_WORLD,
          1,
          0,
          PETSC_DECIDE,
          NULL,
          &(gctx.rho_global_subcomm[i]));CHKERRQ(ierr);
    }

    ierr = PetscObjectSetName((PetscObject)(gctx.phi_global_subcomm[i]),
        std::string("phi_global_subcomm_" + std::to_string(i) + "_on_gctx").c_str());CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject)(gctx.rho_global_subcomm[i]),
        std::string("rho_global_subcomm_" + std::to_string(i) + "_on_gctx").c_str());CHKERRQ(ierr);

  }
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);

  /* Restore local vector arrays */
  ierr = VecRestoreArrayReadAndMemType(sctx.phi_subcomm, &d_phi_val);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(sctx.rho_subcomm, &d_rho_val);CHKERRQ(ierr);

  if (gctx.debug){
    for (PetscInt i=0; i<gctx.nsubcomms; i++){
      ierr = VecGetSize(gctx.rho_global_subcomm[i], &size);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,
          "Hello! the size of the global alias of subcomm vec with index %d is %d\n",
          i,
          size);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}


/* --------------------------------------------------------------------- */
/*!
  Initialize subcomm aliases of local vectors
  \param  lctx - local context
  \param  sctx - subcomm context
  \param  gctx - global context
  \return ierr - PetscErrorCode
  */
PetscErrorCode init_subcomm_local_aliases(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx){

  PetscInt           phi_localsize;   /* size of vector on this MPI rank */
  PetscInt           rho_localsize;   /* size of vector on this MPI rank */
  PetscScalar const  *d_phi_val;      /* read-only pointer to vector's contents on device */
  PetscScalar const  *d_rho_val;      /* read-only pointer to vector's contents on device */
  PetscMemType       mtype_phi;
  PetscMemType       mtype_rho;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;

  /* Get size of local vector */
  ierr = VecGetLocalSize(lctx.seqphi, &phi_localsize);CHKERRQ(ierr);
  ierr = VecGetLocalSize(lctx.seqrho, &rho_localsize);CHKERRQ(ierr);

  /* Get access to local vector */
  ierr = VecGetArrayReadAndMemType(lctx.seqphi, &d_phi_val, &mtype_phi);CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(lctx.seqrho, &d_rho_val, &mtype_rho);CHKERRQ(ierr);

  /* consistency check */
  if (gctx.debug) {
    if (mtype_phi != mtype_rho) SETERRQ(PETSC_COMM_WORLD,
        PETSC_ERR_ARG_NOTSAMETYPE,
        "phi and rho vectors don't have the memory type!");
    if (!gctx.VecCreate_type_WithArray) SETERRQ(PETSC_COMM_WORLD,
        PETSC_ERR_ARG_BADPTR,
        "the function pointer in gctx is invalid!");
  }

  /* Create subcomm aliases of local vectors */
  ierr = gctx.VecCreate_type_WithArray(PetscObjectComm(PetscObject(sctx.phi_subcomm)),
      1,
      phi_localsize,
      PETSC_DECIDE,
      d_phi_val,
      &sctx.phi_subcomm_local);CHKERRQ(ierr);
  ierr = gctx.VecCreate_type_WithArray(PetscObjectComm(PetscObject(sctx.phi_subcomm)),
      1,
      rho_localsize,
      PETSC_DECIDE,
      d_rho_val,
      &sctx.rho_subcomm_local);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(sctx.phi_subcomm_local), "phi_subcomm_local_on_sctx");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(sctx.rho_subcomm_local), "rho_subcomm_local_on_sctx");CHKERRQ(ierr);

  /* Restore local vector arrays */
  ierr = VecRestoreArrayReadAndMemType(lctx.seqphi, &d_phi_val);CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(lctx.seqrho, &d_rho_val);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

