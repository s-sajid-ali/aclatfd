/*
 * Poisson Eq in 3D : Laplacian phi = -rho/eps0
 * Domain: -Lx/y/z < x,y,z < Lx/y/z ;
 * with dirichlet boundary conditions, 0 at all boundaries
 * SI units are used
 * Note that the scaled equation is implemented,
 * as it reduces the condition number of the matrix,
 * scaling factor = area of unit cell = 1/(Lx*Ly*Lz)
 */

static char help[] = "Solves 3D Poisson \n\n";

#include "poisson3d.h"

int main(int argc, char **argv) {
  AppCtx appctx;            /* user-defined application context */
  KSP ksp;                  /* krylov solver */
  PetscInt it;              /* index */
  PetscViewer rhoviewer;    /* hdf5 file viewer for rho */
  PetscViewer phiviewer;    /* hdf5 file viewer for phi */
  PetscLogStage solvestage; /* log stage for logging linear solve */
  PetscErrorCode ierr;      /* Error code */

  ierr = PetscInitialize(&argc, &argv, (char *)0, help);
  if (ierr)
    return ierr;
  ierr = PetscLogStageRegister("linear-solve", &solvestage);
  CHKERRQ(ierr);

  appctx.Lx = 1e-3;
  appctx.Ly = 1e-3;
  appctx.Lz = 1e-3;

  ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                      DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 256, 256, 1024,
                      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL,
                      NULL, NULL, &appctx.da);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);
  CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);
  CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(appctx.da, -appctx.Lx, appctx.Lx, -appctx.Ly,
                                   appctx.Ly, -appctx.Lz, appctx.Lz);
  CHKERRQ(ierr);

  /* create discretization matrix */
  ierr = DMCreateMatrix(appctx.da, &(appctx.A));
  CHKERRQ(ierr);

  /* create rho and phi vectors */
  ierr = DMCreateGlobalVector(appctx.da, &appctx.rho);
  CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.rho, "rho_vec");
  CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(appctx.da, &appctx.phi);
  CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.phi, "phi_vec");
  CHKERRQ(ierr);
  ierr = VecSet(appctx.phi, 0.0);
  CHKERRQ(ierr);

  /* fill matrix */
  ierr = ComputeMatrix(&appctx);
  CHKERRQ(ierr);

  /* fill rho vector */
  ierr = ComputeRHSGaussian(&appctx);
  CHKERRQ(ierr);
  // ierr = ComputeRHSPoint(&appctx);CHKERRQ(ierr);

  /* SI units for Poisson eq. */
  appctx.eps0 = 8.85418781281e-12;
  ierr = VecScale(appctx.rho, (1 / appctx.eps0));
  CHKERRQ(ierr);

  /* Scaling factor of hx*hy*hz */
  ierr = VecScale(appctx.rho, 1 / (appctx.Lx * appctx.Ly * appctx.Lz));
  CHKERRQ(ierr);

  /* create krylov solver */
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
  CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, appctx.A, appctx.A);
  CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);
  CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);
  CHKERRQ(ierr);
  ierr = KSPSetDM(ksp, appctx.da);
  CHKERRQ(ierr);

  /* solve - warm up */
  ierr = KSPSolve(ksp, appctx.rho, appctx.phi);
  CHKERRQ(ierr);
  /* reuse preconditioner in subsequent solves */
  ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE);
  CHKERRQ(ierr);

  PetscLogStagePush(solvestage);
  CHKERRQ(ierr);
  for (it = 0; it < 5; it++) {
    ierr = KSPSolve(ksp, appctx.rho, appctx.phi);
    CHKERRQ(ierr);
  }
  PetscLogStagePop();
  CHKERRQ(ierr);

  /* Scaling factor of hx*hy*hz */
  ierr = VecScale(appctx.rho, (appctx.Lx * appctx.Ly * appctx.Lz));
  CHKERRQ(ierr);
  ierr = VecScale(appctx.phi, (appctx.Lx * appctx.Ly * appctx.Lz));
  CHKERRQ(ierr);

  /* write rho and phi to hdf5 files */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "rho.h5", FILE_MODE_WRITE,
                             &rhoviewer);
  CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "phi.h5", FILE_MODE_WRITE,
                             &phiviewer);
  CHKERRQ(ierr);

  ierr = VecView(appctx.rho, rhoviewer);
  CHKERRQ(ierr);
  ierr = VecView(appctx.phi, phiviewer);
  CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&rhoviewer);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&phiviewer);
  CHKERRQ(ierr);

  /* cleanup */
  ierr = KSPDestroy(&ksp);
  CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.phi);
  CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.rho);
  CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.A);
  CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);
  CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
