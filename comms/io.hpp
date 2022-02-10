#ifndef IO_H
#define IO_H

#include <petscviewerhdf5.h>
#include "comms.hpp"


PetscErrorCode save_state(LocalCtx& lctx, SubcommCtx& sctx, GlobalCtx& gctx, std::string& name);

#endif
