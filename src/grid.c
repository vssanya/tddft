#include "grid.h"
#include <math.h>

size_t grid2_size(grid2_t const* grid) {
	return grid->n[0]*grid->n[1];
}

size_t grid3_size(grid3_t const* grid) {
	return grid->n[0]*grid->n[1]*grid->n[2];
}

__attribute__((pure))
	double sp_grid_r(sp_grid_t const* grid, int ir) {
		check_index(grid, iR, ir);
		return grid->d[iR]*(ir+1);
	}

__attribute__((pure))
	int sp_grid_ir(sp_grid_t const* grid, double r) {
		return (int) (r/grid->d[iR]) - 1;
	}

__attribute__((pure))
	double sp_grid_c(sp_grid_t const* grid, int ic) {
		check_index(grid, iC, ic);
		return grid->d[iC]*ic - 1.0;
	}

__attribute__((pure))
	int sp_grid_ic(sp_grid_t const* grid, double c) {
		return (int) ((c + 1.0) / grid->d[iC]);
	}

double sp_grid_phi(sp_grid_t const* grid, int ip) {
	check_index(grid, iP, ip);
	return grid->d[iP]*ip;
}

double sh_grid_r(sh_grid_t const* grid, int ir) {
	check_index(grid, iR, ir);
	return grid->d[iR]*(ir+1);
}

double sh_grid_r_max(sh_grid_t const* grid) {
	return sh_grid_r(grid, grid->n[iR]-1);
}

int sh_grid_l(sh_grid_t const* grid, int il) {
	check_index(grid, iL, il);
	return il;
}

int sh_grid_m(sh_grid_t const* grid, int im) {
	assert(im >= -grid->n[iL] && im <= grid->n[iL]);
	return im;
}

#define check_index(grid, space, index) assert(index >= 0 || index < grid->n[space])

size_t grid2_index(grid2_t const* grid, int i[2]) {
	check_index(grid, 0, i[0]);
	check_index(grid, 1, i[1]);
	return i[0] +
		i[1]*grid->n[0];
}

size_t grid3_index(grid3_t const* grid, int i[3]) {
	check_index(grid, 0, i[0]);
	check_index(grid, 1, i[1]);
	check_index(grid, 2, i[2]);
	return i[0] + 
		i[1]*grid->n[0] +
		i[2]*grid->n[1]*grid->n[0];
}

sp_grid_t* sp_grid_new(int n[3], double r_max) {
	sp_grid_t* grid = malloc(sizeof(sp_grid_t));

	for (int i = 0; i < 3; ++i) {
		grid->n[i] = n[i];
	}

	grid->d[iR] = r_max/n[iR];
	grid->d[iC] = 2.0/(n[iC]-1);
	grid->d[iP] = 2*M_PI/n[iP];

	return grid;
}

sh_grid_t* sh_grid_new(int n[2], double r_max) {
	sh_grid_t* grid = malloc(sizeof(sh_grid_t));

	for (int i = 0; i < 2; ++i) {
		grid->n[i] = n[i];
	}

	grid->d[iR] = r_max/n[iR];
	grid->d[iL] = 1;

	return grid;
}

grid2_t* ct_grid_new(int n[2], double x_max, double y_max) {
	grid2_t* grid = (grid2_t*) malloc(sizeof(grid2_t));

	for (int i = 0; i < 2; ++i) {
		grid->n[i] = n[i];
	}

	grid->d[iX] = (2*x_max)/(n[iX]-1);
	grid->d[iY] = y_max/n[iY];

	return grid;
}

double ct_grid_x(grid2_t const* grid, int ix) {
	return 0.5*grid->d[iX]*(2*ix - grid->n[iX] + 1);
}

double ct_grid_y(grid2_t const* grid, int iy) {
	return iy*grid->d[iY];
}

void sh_grid_del(sh_grid_t* grid) {
	free(grid);
}

void sp_grid_del(sp_grid_t* grid) {
	free(grid);
}
