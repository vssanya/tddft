/*
 * =====================================================================================
 *
 *       Filename:  sae_workspace.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01.11.2017 16:41:16
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#pragma once

#include "wf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	ws_wf_t* ws_wf;
} ws_sae_t;

ws_sae_t* ws_sae_new(sh_grid_t const* grid, uabs_sh_t const* uabs, int num_threads);

#ifdef __cplusplus
}
#endif
