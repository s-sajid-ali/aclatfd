#include <petscksp.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>

/* Linear solver struct */
typedef struct{

  DM           da;   /* DMDA to manage grid and vecs */
  Mat          A;    /* discretization matrix */
  Vec          rho;  /* charge density */
  Vec          phi;  /* electric potential */
  PetscReal    Lx;   /* length along x */
  PetscReal    Ly;   /* length along x */
  PetscReal    Lz;   /* length along x */
  PetscScalar  eps0; /* permittivity of free space, SI units! */

} AppCtx;


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
