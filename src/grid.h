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

inline size_t grid2_size(grid2_t const* grid) {
	return grid->n[0]*grid->n[1];
}

inline size_t grid3_size(grid3_t const* grid) {
	return grid->n[0]*grid->n[1]*grid->n[2];
}

#define check_index(grid, space, index) assert(index >= 0 || index < grid->n[space])

inline size_t grid2_index(grid2_t const* grid, int i[2]) {
    check_index(grid, 0, i[0]);
    check_index(grid, 1, i[1]);
    return i[0] +
           i[1]*grid->n[0];
}

inline size_t grid3_index(grid3_t const* grid, int i[3]) {
    check_index(grid, 0, i[0]);
    check_index(grid, 1, i[1]);
    check_index(grid, 2, i[2]);
    return i[0] + 
           i[1]*grid->n[0] +
           i[2]*grid->n[1]*grid->n[0];
}

sp_grid_t* sp_grid_new(int n[3], double r_max);
void sp_grid_del(sp_grid_t* grid);

sh_grid_t* sh_grid_new(int n[2], double r_max);
void sh_grid_del(sh_grid_t* grid);

__attribute__((pure)) inline
double sp_grid_r(sp_grid_t const* grid, int ir) {
    check_index(grid, iR, ir);
    return grid->d[iR]*(ir+1);
}

__attribute__((pure)) inline
int sp_grid_ir(sp_grid_t const* grid, double r) {
	return (int) (r/grid->d[iR]) - 1;
}

__attribute__((pure)) inline
double sp_grid_c(sp_grid_t const* grid, int ic) {
    check_index(grid, iC, ic);
    return grid->d[iC]*ic - 1.0;
}

__attribute__((pure)) inline
int sp_grid_ic(sp_grid_t const* grid, double c) {
	return (int) ((c + 1.0) / grid->d[iC]);
}

inline double sp_grid_phi(sp_grid_t const* grid, int ip) {
    check_index(grid, iP, ip);
    return grid->d[iP]*ip;
}

inline double sh_grid_r(sh_grid_t const* grid, int ir) {
    check_index(grid, iR, ir);
    return grid->d[iR]*(ir+1);
}

inline double sh_grid_r_max(sh_grid_t const* grid) {
	return sh_grid_r(grid, grid->n[iR]-1);
}

inline int sh_grid_l(sh_grid_t const* grid, int il) {
    check_index(grid, iL, il);
    return il;
}

inline int sh_grid_m(sh_grid_t const* grid, int im) {
    assert(im >= -grid->n[iL] && im <= grid->n[iL]);
    return im;
}

#ifdef __cplusplus
}
#endif
