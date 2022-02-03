#ifndef UTILS_H
#define UTILS_H

#include "comms.hpp"

PetscErrorCode init_solversubcomms(SubcommCtx& sctx, GlobalCtx& gctx);

PetscErrorCode init_localvecs(LocalCtx& lctx, GlobalCtx& gctx);
PetscErrorCode init_subcommvecs(SubcommCtx& sctx, GlobalCtx& gctx);

PetscErrorCode init_global_local_aliases(LocalCtx& lctx, GlobalCtx& gctx);
PetscErrorCode init_global_subcomm_aliases(SubcommCtx& sctx, GlobalCtx& gctx);
PetscErrorCode init_subcomm_local_aliases(LocalCtx& lctx, SubcommCtx& sctx);

PetscErrorCode init_global_subcomm_scatters(SubcommCtx& sctx, GlobalCtx& gctx);
PetscErrorCode init_subcomm_local_scatters(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx);

PetscErrorCode finalize(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx);

#endif
