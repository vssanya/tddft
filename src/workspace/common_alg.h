#pragma once

#include "../sh_wavefunc.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif

void wf_prop_ang_l(sh_wavefunc_t* wf, cdouble dt, int l, int l1, sh_f Ul);

#ifdef __cplusplus
}
#endif
