#ifndef ALIAS_H
#define ALIAS_H

#include "comms.hpp"


PetscErrorCode determine_veccreatewitharray_func(GlobalCtx& gctx);


PetscErrorCode init_global_local_aliases(LocalCtx& lctx, GlobalCtx& gctx);

PetscErrorCode init_global_subcomm_aliases(SubcommCtx& sctx, GlobalCtx& gctx);

PetscErrorCode init_subcomm_local_aliases(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx);

#endif
