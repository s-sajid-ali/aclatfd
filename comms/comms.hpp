#ifndef COMMS_H
#define COMMS_H

#include <petscvec.h>

#include <functional>
#include <vector>
#include <string>

/* Local (per MPI rank) context */
struct LocalCtx {

  Vec   seqphi;                            /*! local seq vector */
  Vec   seqrho;                            /*! local seq vector */

};


/* Subcomm context */
struct SubcommCtx {

  Vec           phi_subcomm_local;         /*! subcomm alias of the local vector on each MPI rank */
  Vec           rho_subcomm_local;         /*! subcomm alias of the local vector on each MPI rank */
  Vec           phi_subcomm;               /*! vector on the subcomm */
  Vec           rho_subcomm;               /*! vector on the subcomm */

  VecScatter    scat_subcomm_to_local;     /*! VecScatter from subcomm vector to constituent local vectors */
  IS            ix_scat_subcomms_to_local; /*! IndexSet for scatter from subcomm vector to constituent local vectors */
  IS            iy_scat_subcomms_to_local; /*! IndexSet for scatter from subcomm vector to constituent local vectors */

  PetscSubcomm  solverpsubcomm;            /*! PETSc subcommunicator for this concurrent solver instance */
  MPI_Comm      solversubcomm;             /*! MPI subcommunicator for this concurrent solver instance */

  PetscMPIInt   solversubcommid;           /*! a unique Int on each solver subcomm (max value of global ranks in subcomm) */
  PetscMPIInt   solver_rank;               /*! solver subcomm MPI communicator rank */
  PetscMPIInt   solver_size;               /*! solver subcomm MPI communicator size */

};


/* Global (over all MPI ranks) context */
struct GlobalCtx {

  PetscInt       nsubcomms = 2;               /*! total number of subcomms */
  PetscInt       nsize = 12;                  /*! the size of the problem, which is size of vectors on each MPI rank */
  PetscBool      debug = PETSC_FALSE;         /*! enable verbose outputs */
  PetscMPIInt    global_rank;                 /*! global MPI communicator rank */
  PetscMPIInt    global_size;                 /*! global MPI communicator size */
  PetscLogStage  logstages[5];                /*! stages for logging performance */

  Vec            phi_global_local;            /*! global alias of the local vector on each MPI rank */
  Vec            rho_global_local;            /*! global alias of the local vector on each MPI rank */

  std::vector<Vec> phi_global_subcomm;        /*! array of global alias of the vector on each subcomm */
  std::vector<Vec> rho_global_subcomm;        /*! array of global alias of the vector on each subcomm */

  std::vector<VecScatter> scat_glocal_to_subcomms;    /*! array of VecScatter(s) from global alias of local vectors to all subcomm vectors */
  IS                      ix_scat_glocal_to_subcomms; /*! IndexSet for scatters from global alias of local vectors to all subcomm vectors */
  IS                      iy_scat_glocal_to_subcomms; /*! IndexSet for scatters from global alias of local vectors to all subcomm vectors */

  const std::string        vectype="kokkos";  /*! type of all vectors */
  std::vector<PetscMPIInt> sids;              /*! holds the solversubcommmid's from each solver */

  std::function<
    PetscErrorCode(MPI_Comm,PetscInt, PetscInt,PetscInt,const PetscScalar *,Vec *)
    >   VecCreate_type_WithArray;             /*! function pointer to hold the appropriate function for creating vectors with array */

};

#endif
