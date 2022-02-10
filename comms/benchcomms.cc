/*
 * A demo version of the communication pattern required by the eventual
 * implementation of a finite-difference based 3D space charge solver
 * for use in the synergia2-devel3 PIC code.
 *
 * This program does the following:
 *  [1] scatters a rho vector from all MPI ranks to all solver-subcomms
 *  [2] scatters a phi vector to all MPI ranks from each subcomm, concurrently.
 *
 */

static char help[] = "Benchmark communications space charge 3D solver! \n\n";

#include "alias.hpp"
#include "utils.hpp"
#include "io.hpp"

#include <mpi.h>

int main(int argc,char **argv){

  PetscErrorCode    ierr;          /* Error code */
  LocalCtx          lctx;          /* Local context on each MPI rank */
  SubcommCtx        sctx;          /* Subcomm context on each subcomm */
  GlobalCtx         gctx;          /* Global context on all subcomms */
  std::string       filename;      /* filename for dumping global members */
  std::string       stagename;     /* name for logging stage */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize PETSc and MPI, query mpi rank and size
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  gctx.nsubcomms = 2;
  ierr = PetscOptionsGetInt(NULL, NULL, "-nsubcomms", &gctx.nsubcomms, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &gctx.global_rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &gctx.global_size);CHKERRMPI(ierr);
  /* Exit immediately if number of subcomms > MPI ranks! */
  if (gctx.global_size < gctx.nsubcomms) SETERRQ(PETSC_COMM_WORLD,
      PETSC_ERR_USER_INPUT,
      "Need to use at least as many subcomms as MPI ranks!");
  ierr = PetscOptionsGetBool(NULL,NULL, "-debug", &gctx.debug, NULL);CHKERRQ(ierr);

  /* Register logging stages */
  ierr = PetscLogStageRegister(const_cast<char*>("3ds-background"), &gctx.logstages[0]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister(const_cast<char*>("3ds g-to-s scat"), &gctx.logstages[1]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister(const_cast<char*>("3ds pde-solve"), &gctx.logstages[2]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister(const_cast<char*>("3ds s-to-l scat"), &gctx.logstages[3]);CHKERRQ(ierr);

  /* Initialize task subcomms, display task-subcomm details */
  ierr = init_solversubcomms(sctx, gctx);CHKERRQ(ierr);

  /* Local rho and phi vectors on each MPI rank */
  ierr = init_localvecs(lctx, gctx);CHKERRQ(ierr);

  /* determine the appropriate function to create vector with array */
  ierr = determine_veccreatewitharray_func(gctx);CHKERRQ(ierr);

  /* rho and phi vectors on each subcomm */
  ierr = init_subcommvecs(sctx, gctx);CHKERRQ(ierr);

  /* create global aliases of local vectors */
  ierr = init_global_local_aliases(lctx, gctx);CHKERRQ(ierr);

  /* create global aliases of subcomm vectors */
  ierr = init_global_subcomm_aliases(sctx, gctx);CHKERRQ(ierr);

  /* create subcomm aliases of local vectors */
  ierr = init_subcomm_local_aliases(lctx, sctx, gctx);CHKERRQ(ierr);

  /* Initialize rho and phi on each rank */
  ierr = VecSet(lctx.seqrho, ((1.0)/(gctx.global_size)));CHKERRQ(ierr);
  ierr = VecSet(lctx.seqphi, 0.0);CHKERRQ(ierr);

  /* Initialize global (alias of local) to subcomm scatters */
  ierr = init_global_subcomm_scatters(sctx, gctx);CHKERRQ(ierr);

  /* Initialize subcomm (alias of local) to local scatters */
  ierr = init_subcomm_local_scatters(lctx, sctx, gctx);CHKERRQ(ierr);

  /* Save current state of global members to file */
  filename = "prescatters.h5";
  ierr = save_state(lctx, sctx, gctx, filename);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup complete, benchmark begins here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscLogStagePush(gctx.logstages[0]);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     global to subcomm scatters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscLogStagePush(gctx.logstages[1]);CHKERRQ(ierr);
  /* Begin global (alias of local) to subcomm scatters! */
  for (PetscInt i=0; i<gctx.nsubcomms; i++) {

    ierr = VecScatterBegin(gctx.scat_glocal_to_subcomms[i],
        gctx.rho_global_local,
        gctx.rho_global_subcomm[i],
        ADD_VALUES,
        SCATTER_FORWARD);CHKERRQ(ierr);

  }

  /* Hopefully there is some unrelated work that can occur here! */

  /* End global (alias of local) to subcomm scatters! */
  for (PetscInt i=0; i<gctx.nsubcomms; i++) {

    ierr = VecScatterEnd(gctx.scat_glocal_to_subcomms[i],
        gctx.rho_global_local,
        gctx.rho_global_subcomm[i],
        ADD_VALUES,
        SCATTER_FORWARD);CHKERRQ(ierr);

  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     concurrent operations on each solver subcommunicator
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscLogStagePush(gctx.logstages[2]);CHKERRQ(ierr);
  /* The value of all elements in sctx->rho_subcomm should be 1 */
  ierr = VecScale(sctx.rho_subcomm, ((1.0)/(gctx.nsubcomms)));CHKERRQ(ierr);
  ierr = VecCopy(sctx.rho_subcomm, sctx.phi_subcomm);CHKERRQ(ierr);
  /* The value of all elements in sctx->phi_subcomm should be 1/nsubcomms */
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     subcomm to local scatters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscLogStagePush(gctx.logstages[3]);CHKERRQ(ierr);
  /* Begin subcomm (alias of local) to local scatters! */
  ierr = VecScatterBegin(sctx.scat_subcomm_to_local,
      sctx.phi_subcomm,
      sctx.phi_subcomm_local,
      INSERT_VALUES,
      SCATTER_REVERSE);CHKERRQ(ierr);

  /* Hopefully there is some unrelated work that can occur here! */

  /* End subcomm (alias of local) to local scatters! */
  ierr = VecScatterEnd(sctx.scat_subcomm_to_local,
      sctx.phi_subcomm,
      sctx.phi_subcomm_local,
      INSERT_VALUES,
      SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscLogStagePop();CHKERRQ(ierr); /* for ending 0th logstage */

  /* Save current state of global members to file */
  filename = "postscatters.h5";
  ierr = save_state(lctx, sctx, gctx, filename);CHKERRQ(ierr);

  ierr = finalize(lctx, sctx, gctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
