#include "eigen.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_eigen.h>

#include <lapacke.h>

#include "linalg.h"


eigen_ws_t* eigen_ws_alloc(sh_grid_t const* grid) {
	eigen_ws_t* ws = malloc(sizeof(eigen_ws_t));
	ws->grid = grid;

	ws->evec = malloc(sizeof(double)*grid->n[iL]*grid->n[iR]*grid->n[iR]);
	ws->eval = malloc(sizeof(double)*grid->n[iL]*grid->n[iR]);

	return ws;
}

void eigen_ws_free(eigen_ws_t* ws) {
	free(ws->eval);
	free(ws->evec);
	free(ws);
}

void eigen_calc_dr4(eigen_ws_t* ws, sh_f u, int Z) {
	int const Nr = ws->grid->n[iR];

	gsl_eigen_symmv_workspace* gsl_ws = gsl_eigen_symmv_alloc(Nr);

	gsl_matrix* A = gsl_matrix_calloc(Nr, Nr);

	double const dr = ws->grid->d[iR];
	double const dr2 = dr*dr;

	double const d2[3] = {1.0/dr2, -2.0/dr2, 1.0/dr2};
	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	double const M2[3] = {
		1.0/12.0,
		10.0/12.0,
		1.0/12.0
	};

	const double M2_l0_11 = 1.0 + d2_l0_11*dr2/12.0;

	for (int il=0; il<ws->grid->n[iL]; ++il) {
		if (il == 0) {
			linalg_tdm_inv(M2_l0_11, M2, Nr, A->data);
			linalg_m_dot_tdm(Nr, A->data, -0.5*d2_l0_11, (double[]){-0.5*d2[0], -0.5*d2[1], -0.5*d2[2]});
		} else {
			linalg_tdm_inv(M2[1], M2, Nr, A->data);
			linalg_m_dot_tdm(Nr, A->data, -0.5*d2[1], (double[]){-0.5*d2[0], -0.5*d2[1], -0.5*d2[2]});
		}

		for (int ir = 0; ir < Nr; ++ir) {
			A->data[ir + Nr*ir] += u(ws->grid, ir, il, 0);
		}

		gsl_matrix_view evec = gsl_matrix_view_array(&ws->evec[Nr*Nr*il], Nr, Nr);
		gsl_vector_view eval = gsl_vector_view_array(&ws->eval[Nr*il], Nr);

		gsl_eigen_symmv(A, &eval.vector, &evec.matrix, gsl_ws);
		gsl_eigen_symmv_sort(&eval.vector, &evec.matrix, GSL_EIGEN_SORT_VAL_ASC);
	}

	gsl_matrix_free(A);

	gsl_eigen_symmv_free(gsl_ws);
}

void eigen_calc_for_atom(eigen_ws_t* ws, atom_t const* atom) {
	double u(sh_grid_t const* grid, int ir, int l, int m) {
		double const r = sh_grid_r(grid, ir);
		return l*(l+1)/(2*r*r) + atom->u(atom, grid, ir);
	}

	eigen_calc_dr4(ws, u, atom->Z);
	//eigen_calc(ws, u, atom->Z);
}

void eigen_calc(eigen_ws_t* ws, sh_f u, int Z) {
	int const Nr = ws->grid->n[iR];

	double* lapack_ws = malloc(sizeof(double)*(2*Nr-1));

	double const dr = ws->grid->d[iR];
	double const dr2 = dr*dr;

	double const d2[3] = {1.0/dr2, -2.0/dr2, 1.0/dr2};
	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	double* A_d;
	double* A_d_up = malloc(sizeof(double)*(Nr-1));

	for (int il=0; il<ws->grid->n[iL]; ++il) {
		double* eval = &ws->eval[Nr*il];
		double* evec = &ws->evec[Nr*Nr*il];

		A_d = eval;

		{
			int ir = 0;
			A_d_up[ir]  = -0.5*d2[0];
			if (il != 0) {
				A_d[ir] = -0.5*d2[1] + u(ws->grid, ir, il, 0);
			} else {
				A_d[ir] = -0.5*d2_l0_11 + u(ws->grid, ir, il, 0);
			}
		}

		for (int ir=1; ir<Nr-1; ++ir) {
			A_d_up[ir] = -0.5*d2[0];
			A_d[ir]    = -0.5*d2[1] + u(ws->grid, ir, il, 0);
		}

		{
			int ir = Nr-1;
			A_d[ir]    = -0.5*d2[1] + u(ws->grid, ir, il, 0);
		}

		LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', Nr, A_d, A_d_up, evec, Nr);
	}

	free(A_d_up);
	free(lapack_ws);
}

int eigen_get_n_with_energy(eigen_ws_t const* ws, double energy) {
	for (int ie = 0; ie<ws->grid->n[iR]; ++ie) {
		if (ws->eval[ie] > energy) {
			return ie;
		}
	}

	return ws->grid->n[iR];
}

//void eigen_calc_dr4(eigen_ws_t* ws, sh_f u, int Z) {
//	int const Nr = ws->grid->n[iR];
//
//	gsl_eigen_genv_workspace* gsl_ws = gsl_eigen_genv_alloc(Nr);
//
//	gsl_matrix* A = gsl_matrix_calloc(Nr, Nr);
//	gsl_matrix* B = gsl_matrix_calloc(Nr, Nr);
//
//	double const dr = ws->grid->d[iR];
//	double const dr2 = dr*dr;
//
//	double const d2[3] = {1.0, -2.0, 1.0};
//	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));
//
//	double const M2[3] = {
//		1.0/12.0*dr2,
//		10.0/12.0*dr2,
//		1.0/12.0*dr2
//	};
//
//	const double M2_l0_11 = (1.0 + d2_l0_11/12.0)*dr2;
//
//	for (int il=0; il<ws->grid->n[iL]; ++il) {
//		{
//			int ir = 0;
//			for (int i=1; i<3; ++i) {
//				gsl_matrix_set(A, ir, ir-1+i, -0.5*d2[i] + M2[i]*u(ws->grid, ir-1+i, il, 0));
//				gsl_matrix_set(B, ir, ir-1+i, M2[i]);
//			}
//
//			if (il == 0) {
//				int i = 1;
//				gsl_matrix_set(A, ir, ir-1+i, -0.5*d2_l0_11 + M2_l0_11*u(ws->grid, ir-1+i, il, 0));
//				gsl_matrix_set(B, ir, ir-1+i, M2_l0_11);
//			}
//		}
//
//		for (int ir=1; ir<Nr-1; ++ir) {
//			for (int i=0; i<3; ++i) {
//				gsl_matrix_set(A, ir, ir-1+i, -0.5*d2[i] + M2[i]*u(ws->grid, ir-1+i, il, 0));
//				gsl_matrix_set(B, ir, ir-1+i, M2[i]);
//			}
//		}
//
//		{
//			int ir = Nr-1;
//			for (int i=0; i<2; ++i) {
//				gsl_matrix_set(A, ir, ir-1+i, -0.5*d2[i] + M2[i]*u(ws->grid, ir-1+i, il, 0));
//				gsl_matrix_set(B, ir, ir-1+i, M2[i]);
//			}
//		}
//
//		gsl_matrix_view evec = gsl_matrix_view_array(&ws->evec[Nr*Nr*il], Nr, Nr);
//		gsl_vector_view eval = gsl_vector_view_array(&ws->eval[Nr*il], Nr);
//
//		gsl_eigen_genv(A, B, &eval.vector, &evec.matrix, gsl_ws);
//	}
//
//	gsl_matrix_free(B);
//	gsl_matrix_free(A);
//
//	gsl_eigen_gensymmv_free(gsl_ws);
//}
