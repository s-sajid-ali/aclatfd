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

#include <petscksp.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>

typedef struct{

  DM         da;   /* DMDA to manage grid and vecs */
  Mat        A;    /* discretization matrix */
  Vec        rho;  /* charge density */
  Vec        phi;  /* electric potential */
  PetscReal  Lx;   /* length along x */
  PetscReal  Ly;   /* length along x */
  PetscReal  Lz;   /* length along x */

} AppCtx;

extern PetscErrorCode ComputeMatrix(void*);
extern PetscErrorCode ComputeRHSGaussian(void*);
extern PetscErrorCode ComputeRHSPoint(void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;     /* user-defined application context */
  KSP            ksp;        /* krylov solver */
  PetscInt       it;         /* index */
  PetscScalar    eps0;       /* permittivity of free space */
  PetscViewer    rhoviewer;  /* hdf5 file viewer for rho */
  PetscViewer    phiviewer;  /* hdf5 file viewer for phi */
  PetscLogStage  solvestage; /* log stage for logging linear solve */
  PetscErrorCode ierr;       /* Error code */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscLogStageRegister("linear-solve",&solvestage);CHKERRQ(ierr);

  appctx.Lx = 1e-3;
  appctx.Ly = 1e-3;
  appctx.Lz = 1e-3;

  ierr = DMDACreate3d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
      DMDA_STENCIL_STAR, 64, 64, 64,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
      NULL,NULL,NULL, &appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(appctx.da, -appctx.Lx, appctx.Lx, -appctx.Ly, appctx.Ly, -appctx.Lz, appctx.Lz);CHKERRQ(ierr);

  /* create discretization matrix */
  ierr = DMCreateMatrix(appctx.da, &(appctx.A));CHKERRQ(ierr);

  /* create rho and phi vectors */
  ierr = DMCreateGlobalVector(appctx.da, &appctx.rho);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.rho, "rho_vec");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(appctx.da, &appctx.phi);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.phi, "phi_vec");CHKERRQ(ierr);
  ierr = VecSet(appctx.phi, 0.0);CHKERRQ(ierr);

  /* fill matrix */
  ierr = ComputeMatrix(&appctx);CHKERRQ(ierr);

  /* fill rho vector */
  ierr = ComputeRHSGaussian(&appctx);CHKERRQ(ierr);
  //ierr = ComputeRHSPoint(&appctx);CHKERRQ(ierr);

  /* SI units for Poisson eq. */
  eps0 = 8.85418781281e-12;
  ierr = VecScale(appctx.rho, (1/eps0));CHKERRQ(ierr);

  /* Scaling factor of hx*hy*hz */
  ierr = VecScale(appctx.rho, 1/(appctx.Lx*appctx.Ly*appctx.Lz));CHKERRQ(ierr);

  /* create krylov solver */
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, appctx.A, appctx.A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp, appctx.da);CHKERRQ(ierr);

  /* solve - warm up */
  ierr = KSPSolve(ksp, appctx.rho, appctx.phi);CHKERRQ(ierr);

  PetscLogStagePush(solvestage);CHKERRQ(ierr);
  for (it=0; it<5; it++){
    ierr = KSPSolve(ksp, appctx.rho, appctx.phi);CHKERRQ(ierr);
  }
  PetscLogStagePop();CHKERRQ(ierr);

  /* Scaling factor of hx*hy*hz */
  ierr = VecScale(appctx.rho, (appctx.Lx*appctx.Ly*appctx.Lz));CHKERRQ(ierr);
  ierr = VecScale(appctx.phi, (appctx.Lx*appctx.Ly*appctx.Lz));CHKERRQ(ierr);

  /* write rho and phi to hdf5 files */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"rho.h5",
      FILE_MODE_WRITE,&rhoviewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"phi.h5",
      FILE_MODE_WRITE,&phiviewer);CHKERRQ(ierr);

  ierr = VecView(appctx.rho, rhoviewer);CHKERRQ(ierr);
  ierr = VecView(appctx.phi, phiviewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&rhoviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&phiviewer);CHKERRQ(ierr);

  /* cleanup */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.phi);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.rho);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.A);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeRHSGaussian(void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;  /* user-defined application context */
  DMDALocalInfo  info;                    /* For storing DMDA info */
  PetscInt       i, j, k;
  PetscScalar    ***barray;
  PetscInt       idx, idy, idz;
  PetscScalar    coordsmin[3], coordsmax[3];
  PetscScalar    sx, sy, sz;
  PetscScalar    x, y, z;
  PetscErrorCode ierr;                    /* Error code */

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(appctx->da, &info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, appctx->rho, &barray);CHKERRQ(ierr);
  ierr = DMGetBoundingBox(appctx->da, coordsmin, coordsmax); CHKERRQ(ierr);

  for (k = info.zs; k<info.zs+info.zm; k++) {
    for (j = info.ys; j < info.ys+info.ym; j++) {
      for (i = info.xs; i < info.xs+info.xm; i++) {
        if (i==0 || j==0 || k==0 || i==info.mx-1 || j==info.my-1 || k==info.mz-1) {
          barray[k][j][i] = 0.0;
        } else {

          idx = (i < info.mx/2) ? (PetscInt)(info.mx/2) - i : i - (PetscInt)(info.mx/2);
          idy = (j < info.my/2) ? (PetscInt)(info.my/2) - j : j - (PetscInt)(info.my/2);
          idz = (k < info.mz/2) ? (PetscInt)(info.mz/2) - k : k - (PetscInt)(info.mz/2);

          x = (i < info.mx/2) ? -1*coordsmin[0]*(idx/(PetscReal)info.mx) : coordsmax[0]*(idx/(PetscReal)info.mx);
          y = (j < info.my/2) ? -1*coordsmin[1]*(idy/(PetscReal)info.my) : coordsmax[1]*(idy/(PetscReal)info.my);
          z = (k < info.mz/2) ? -1*coordsmin[2]*(idz/(PetscReal)info.mz) : coordsmax[2]*(idz/(PetscReal)info.mz);

          barray[k][j][i] = PetscExpReal( -1 * ( ((x*x)/(2*(sx*sx))) + ((y*y)/(2*(sy*sy))) + ((z*z)/(2*(sz*sz))) ) );
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(appctx->da, appctx->rho, &barray);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(appctx->rho);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(appctx->rho);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHSPoint(void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;  /* user-defined application context */
  DMDALocalInfo  info;                    /* For storing DMDA info */
  PetscInt       i, j, k;
  PetscScalar    ***barray;
  PetscInt       idx, idy, idz;
  PetscScalar    sx, sy, sz;
  PetscScalar    x, y, z;
  PetscScalar    coordsmin[3], coordsmax[3];
  PetscErrorCode ierr;                    /* Error code */

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(appctx->da, &info); CHKERRQ(ierr);
  ierr = DMGetBoundingBox(appctx->da, coordsmin, coordsmax); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da, appctx->rho, &barray);CHKERRQ(ierr);

  /* std-dev of gaussian */
  sx = coordsmax[0]/10.0;
  sy = coordsmax[1]/10.0;
  sz = coordsmax[2]/100.0;

  for (k = info.zs; k<info.zs+info.zm; k++) {
    for (j = info.ys; j < info.ys+info.ym; j++) {
      for (i = info.xs; i < info.xs+info.xm; i++) {
        if (i==info.mx/2 && j==info.my/2 && k==info.mz/2) {
          barray[k][j][i] = 1.0;
        } else {
          barray[k][j][i] = 0.0;
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(appctx->da, appctx->rho, &barray);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(appctx->rho);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(appctx->rho);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode ComputeMatrix(void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;  /* user-defined application context */
  DMDALocalInfo  info;                    /* For storing DMDA info */
  PetscInt       i, j, k;
  PetscScalar    v[7];
  PetscScalar    hx, hy, hz;
  PetscScalar    hxhydhz, hxdhyhz, dhxhyhz;
  PetscScalar    coordsmin[3], coordsmax[3];
  MatStencil     row,col[7];
  PetscErrorCode ierr;                    /* Error code */

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(appctx->da, &info); CHKERRQ(ierr);
  ierr = DMGetBoundingBox(appctx->da, coordsmin, coordsmax); CHKERRQ(ierr);

  hx = (coordsmax[0] - coordsmin[0]) / (PetscReal)(info.mx);
  hy = (coordsmax[1] - coordsmin[1])  / (PetscReal)(info.my);
  hz = (coordsmax[2] - coordsmin[2]) / (PetscReal)(info.mz);

  hxhydhz = (hx*hy)/hz;
  hxdhyhz = (hx*hz)/hy;
  dhxhyhz = (hy*hz)/hx;

  for (k = info.zs; k<info.zs+info.zm; k++) {
    for (j = info.ys; j < info.ys+info.ym; j++) {
      for (i = info.xs; i < info.xs+info.xm; i++) {
        row.i = i; row.j = j; row.k = k;
        if (i==0 || j==0 || k==0 || i==info.mx-1 || j==info.my-1 || k==info.mz-1) {
          v[0] = 1.0;  // on boundary: trivial equation
          ierr = MatSetValuesStencil(appctx->A, 1, &row, 1, &row, v, INSERT_VALUES);CHKERRQ(ierr);
        } else {
          v[0] = -hxhydhz;                     col[0].i = i;     col[0].j = j;     col[0].k = k-1;
          v[1] = -hxdhyhz;                     col[1].i = i;     col[1].j = j-1;   col[1].k = k;
          v[2] = -dhxhyhz;                     col[2].i = i-1;   col[2].j = j;     col[2].k = k;
          v[3] =  2*(hxhydhz+hxdhyhz+dhxhyhz); col[3].i = row.i; col[3].j = row.j; col[3].k = row.k;
          v[4] = -dhxhyhz;                     col[4].i = i+1;   col[4].j = j;     col[4].k = k;
          v[5] = -hxdhyhz;                     col[5].i = i;     col[5].j = j+1;   col[5].k = k;
          v[6] = -hxhydhz;                     col[6].i = i;     col[6].j = j;     col[6].k = k+1;
          ierr = MatSetValuesStencil(appctx->A, 1, &row, 7, col, v, INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(appctx->A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
