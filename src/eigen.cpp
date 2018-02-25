#include "eigen.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_eigen.h>

#include <lapacke.h>

#include "linalg.h"


eigen_ws_t* eigen_ws_alloc(ShGrid const* grid) {
	eigen_ws_t* ws = new eigen_ws_t;
	ws->grid = grid;

	ws->evec = new double[grid->n[iL]*grid->n[iR]*grid->n[iR]];
	ws->eval = new double[grid->n[iL]*grid->n[iR]];

	return ws;
}

void eigen_ws_free(eigen_ws_t* ws) {
	delete[] ws->eval;
	delete[] ws->evec;
	delete ws;
}

double eigen_eval(eigen_ws_t const* ws, int il, int ie) {
	int const Nr = ws->grid->n[iR];
	return ws->eval[ie + il*Nr];
}

double eigen_evec(eigen_ws_t const* ws, int il, int ir, int ie) {
	int const Nr = ws->grid->n[iR];
	return ws->evec[ie + ir*Nr + il*Nr*Nr];
}

void eigen_calc_dr4(eigen_ws_t* ws, std::function<double(ShGrid const*, int, int, int)> u, int Z) {
	int const Nr = ws->grid->n[iR];

	gsl_eigen_symmv_workspace* gsl_ws = gsl_eigen_symmv_alloc(Nr);

	gsl_matrix* A = gsl_matrix_calloc(Nr, Nr);

	double const dr = ws->grid->d[iR];
	double const dr2 = dr*dr;

	double const d2[3] = {1.0/dr2, -2.0/dr2, 1.0/dr2};
	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	linalg::tdm_t const m_d2 = { -0.5*d2[1], -0.5*d2[1], {-0.5*d2[0], -0.5*d2[1], -0.5*d2[2]}, Nr };
	linalg::tdm_t const m_d2_l0 = { -0.5*d2_l0_11,  -0.5*d2_l0_11, {-0.5*d2[0], -0.5*d2[1], -0.5*d2[2]}, Nr };

//	double const M2[3] = {
//		1.0/12.0,
//		10.0/12.0,
//		1.0/12.0
//	};

	double const M2_l0_11 = 1.0 + d2_l0_11*dr2/12.0;

	linalg::tdm_t const m_M2 = { 10.0/12.0, 10.0/12.0, {1.0/12.0, 10.0/12.0, 1.0/12.0}, Nr };
	linalg::tdm_t const m_M2_l0 = { M2_l0_11,  M2_l0_11, {1.0/12.0, 10.0/12.0, 1.0/12.0}, Nr };


	for (int il=0; il<ws->grid->n[iL]; ++il) {
		if (il == 0) {
			m_M2_l0.inv(A->data);
			m_d2_l0.matrix_dot(A->data);
		} else {
			m_M2.inv(A->data);
			m_d2.matrix_dot(A->data);
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

void eigen_calc_for_atom(eigen_ws_t* ws, AtomCache const* atom_cache) {
    eigen_calc_dr4(ws, [atom_cache](ShGrid const* grid, int ir, int l, int m) {
		double const r = grid->r(ir);
        return l*(l+1)/(2*r*r) + atom_cache->u(ir);
    }, atom_cache->atom.Z);
}

void eigen_calc(eigen_ws_t* ws, sh_f u, int Z) {
	int const Nr = ws->grid->n[iR];

	double* lapack_ws = new double[2*Nr-1];

	double const dr = ws->grid->d[iR];
	double const dr2 = dr*dr;

	double const d2[3] = {1.0/dr2, -2.0/dr2, 1.0/dr2};
	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	double* A_d;
	double* A_d_up = new double[Nr-1];

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

	delete[] A_d_up;
	delete[] lapack_ws;
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
