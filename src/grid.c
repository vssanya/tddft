#include "grid.h"
#include <math.h>

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

void sh_grid_del(sh_grid_t* grid) {
	free(grid);
}

void sp_grid_del(sp_grid_t* grid) {
	free(grid);
}
