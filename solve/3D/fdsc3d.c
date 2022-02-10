/*
 * A demo version of finite-difference based 3D space charge solver
 * for use in synergia2-devel3 PIC code.
 *
 * This program does the following:
 *  [1] creates a rho vector on all MPI ranks & scatters them to multi subcomms
 *  [2] solves the Poisson eq. on all subcomms concurrently
 *  [3] scatters phi to all MPI ranks from each subcomm, concurrently.
 *
 *  This assumes that the problem is large enough to benefit
 *  from a multi-GPU solve. If this is not the case, then
 *  it is better to do an MPI_Allreduce over the rho vectors
 *  and solve concurrently on all GPUs. The threshold of
 *  problem size where each method is better shall be determined
 *  via profiling on perlmutter soon.
 *
 */

static char help[] = "Prototype space charge 3D solver! \n\n";

#include "poisson3d.h"

int main(int argc,char **argv){

  PetscErrorCode ierr;                    /* Error code */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Local rho and phi vectors one each MPI rank */



  ierr = PetscFinalize();
  return ierr;
}

