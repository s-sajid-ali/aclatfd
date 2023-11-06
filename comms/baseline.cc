/*
 * A demo version of the communication pattern currently implemented
 * in synergia2-devel3 (as of Feb 2022) in the 3D space charge solver.
 *
 * This program does the following:
 *  [1] all-ranks : deep-copy particles to host, MPI_Allreduce over
 *                  all MPI ranks, deep-copy particles to device
 *  [2] all-ranks : deep-copy particles to host, MPI_Allreduce over
 *                  MPI subcomm ranks, deep-copy particles to device ;
 *
 */

static char help[] = "Baseline communications for space charge 3D solver! \n\n";

#include <petscvec.h>

int main(int argc, char **argv) {

  PetscErrorCode ierr;               /* Error code */
  PetscMPIInt grank;                 /* Global MPI rank */
  PetscMPIInt gsize;                 /* Global MPI size */
  PetscSubcomm psubcomm;             /* PETSc sub-communicator */
  MPI_Comm subcomm;                  /* MPI sub-communicator */
  PetscInt nsubcomms = 4;            /* number of sub-communicators */
  PetscInt nsize = 256 * 256 * 1024; /* size of vectors on each rank */
  Vec seqrho;                        /* seq vector */
  Vec seqphi;                        /* seq vector */
  PetscScalar *h_phi_val;            /* pointer to vector's contents on host */
  PetscScalar *h_rho_val;            /* pointer to vector's contents on host */
  PetscScalar const
      *d_phi_val; /* read-only pointer to vector's contents on device */
  PetscScalar const
      *d_rho_val;         /* read-only pointer to vector's contents on device */
  PetscMemType mtype_rho; /* memory type for the rho vector */
  PetscMemType mtype_phi; /* memory type for the phi vector */
  PetscLogStage logstages[2]; /*! stages for logging performance */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize PETSc and MPI, query mpi rank and size
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr)
    return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &grank);
  CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &gsize);
  CHKERRMPI(ierr);

  ierr = PetscLogStageRegister(const_cast<char *>("world-allreduce"),
                               &logstages[0]);
  CHKERRQ(ierr);
  ierr = PetscLogStageRegister(const_cast<char *>("scomm-allreduce"),
                               &logstages[1]);
  CHKERRQ(ierr);

  ierr = PetscSubcommCreate(PETSC_COMM_WORLD, &(psubcomm));
  CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(psubcomm, nsubcomms);
  CHKERRQ(ierr);
  ierr = PetscSubcommSetType(psubcomm, PETSC_SUBCOMM_CONTIGUOUS);
  CHKERRQ(ierr);
  ierr = PetscSubcommGetChild(psubcomm, &(subcomm));
  CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_SELF, &seqrho);
  CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &seqphi);
  CHKERRQ(ierr);
  ierr = VecSetSizes(seqrho, PETSC_DECIDE, nsize);
  CHKERRQ(ierr);
  ierr = VecSetSizes(seqphi, PETSC_DECIDE, nsize);
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(seqrho);
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(seqphi);
  CHKERRQ(ierr);

  /* Initialize rho and phi on each rank to random values */
  ierr = VecSetRandom(seqrho, NULL);
  CHKERRQ(ierr);
  ierr = VecSetRandom(seqphi, NULL);
  CHKERRQ(ierr);

  /* Get access to local vector on device, makes vector reside on device */
  ierr = VecGetArrayReadAndMemType(seqrho, &d_rho_val, &mtype_rho);
  CHKERRQ(ierr);
  ierr = VecGetArrayReadAndMemType(seqphi, &d_phi_val, &mtype_phi);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(seqrho, &d_rho_val);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(seqphi, &d_phi_val);
  CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     First all-reduce over global communicator!
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* No warm phase is needed as we do not use persistent collectives
   * which have been introduced in MPI-4! */

  ierr = PetscLogStagePush(logstages[0]);
  CHKERRQ(ierr);

  /* Get access to local vector on host, makes vector reside on host */
  ierr = VecGetArray(seqrho, &h_rho_val);
  CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, h_rho_val, nsize, MPIU_SCALAR, MPI_SUM,
                        PETSC_COMM_WORLD);
  CHKERRMPI(ierr);
  ierr = VecRestoreArray(seqrho, &h_rho_val);
  CHKERRQ(ierr);

  /* Get access to local vector on device, makes vector reside on device */
  ierr = VecGetArrayReadAndMemType(seqrho, &d_rho_val, &mtype_rho);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(seqrho, &d_rho_val);
  CHKERRQ(ierr);

  ierr = PetscLogStagePop();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Second all-reduce over sub-communicator!
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscLogStagePush(logstages[1]);
  CHKERRQ(ierr);

  /* Get access to local vector on host, makes vector reside on host */
  ierr = VecGetArray(seqphi, &h_phi_val);
  CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE, h_phi_val, nsize, MPIU_SCALAR, MPI_SUM,
                        subcomm);
  CHKERRMPI(ierr);
  ierr = VecRestoreArray(seqphi, &h_phi_val);
  CHKERRQ(ierr);

  /* Get access to local vector on device, makes vector reside on device */
  ierr = VecGetArrayReadAndMemType(seqphi, &d_phi_val, &mtype_phi);
  CHKERRQ(ierr);
  ierr = VecRestoreArrayReadAndMemType(seqphi, &d_phi_val);
  CHKERRQ(ierr);

  ierr = PetscLogStagePop();

  /* cleanup */
  ierr = VecDestroy(&seqrho);
  CHKERRQ(ierr);
  ierr = VecDestroy(&seqphi);
  CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&psubcomm);
  CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
