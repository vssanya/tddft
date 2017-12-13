#pragma once

#include <stdlib.h>
#include <assert.h>


#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Grid for 2D space
 * */
typedef struct {
	int    n[2]; //!< is counts of points
	double d[2]; //!< is steps
} grid2_t;

/*!
 * \brief Grid for 3D space
 * */
typedef struct {
	int    n[3]; //!< is counts of points
	double d[3]; //!< is steps
} grid3_t;

/*!
 * \brief Indexes for coordinate
 * */
enum {
	iX = 0, //!< is index for x
	iY = 1, //!< is index for y
	iZ = 2, //!< is index for z
};

typedef grid3_t sp_grid_t;
/*!
 * \brief Indexes for spherical coordinate
 * */
enum {
	iR = 0, //!< is index for r
	iC = 1, //!< is index for \f$\cos\theta\f$
	iP = 2, //!< is index for \f$\phi\f$
};

typedef grid2_t sh_grid_t;
/*!
 * \brief Indexes for spherical harmonics
 * */
enum {
	iL = 1, //!< is index for l
	iM = 2, //!< is index for m
};

size_t grid2_size(grid2_t const* grid);
size_t grid3_size(grid3_t const* grid);

#define check_index(grid, space, index) assert(index >= 0 || index < grid->n[space])

size_t grid2_index(grid2_t const* grid, int i[2]);
size_t grid3_index(grid3_t const* grid, int i[3]);

sp_grid_t* sp_grid_new(int n[3], double r_max);
void sp_grid_del(sp_grid_t* grid);

sh_grid_t* sh_grid_new(int n[2], double r_max);
void sh_grid_del(sh_grid_t* grid);

__attribute__((pure)) double sp_grid_r(sp_grid_t const* grid, int ir);
__attribute__((pure)) int sp_grid_ir(sp_grid_t const* grid, double r);
__attribute__((pure)) double sp_grid_c(sp_grid_t const* grid, int ic);
__attribute__((pure)) int sp_grid_ic(sp_grid_t const* grid, double c);
double sp_grid_phi(sp_grid_t const* grid, int ip);
double sh_grid_r(sh_grid_t const* grid, int ir);
double sh_grid_r_max(sh_grid_t const* grid);
int sh_grid_l(sh_grid_t const* grid, int il);
int sh_grid_m(sh_grid_t const* grid, int im);

#ifdef __cplusplus
}
#endif
