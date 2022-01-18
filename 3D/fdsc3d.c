/*
 * A demo version of finite-difference based 3D space charge solver
 * for use in synergia2-devel3 PIC code.
 *
 * This program does the following:
 *  [1] creates a rho vector on all MPI ranks
 *  [2] combines them onto a global vector via sum reduction
 *  [3] scatters them to multi subcomms
 *  [4] solves the Poisson eq. on all subcomms concurrently
 *  [5] scatters phi to all MPI ranks from each subcomm, concurrently.
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

#include <petscksp.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>


int main(int argc,char **argv){

  PetscErrorCode ierr;                    /* Error code */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;



  ierr = PetscFinalize();
  return ierr;
}

