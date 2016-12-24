#include "hartree_potential.h"

//typedef double (*_func_t)(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax);
//inline double _l0(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax) {
//	return swf_get_abs_2(wf, ir, il) / rmax;
//}
//
//inline double _integrate(sphere_wavefunc_t const* wf, int ir, int il, _func_t func) {
//	double res = 0.0;
//	double const r = (ir+1)*wf->grid->dr;
//
//	// int_r' then r' < r
//	for (int ir1 = 0; ir1 < ir; ++ir1) {
//		double const r1 = (ir1+1)*wf->grid->dr;
//		res += func(wf, ir1, il, r1, r);
//	}
//
//	// int_r' then r' > r
//	for (int ir1 = ir; ir1 < wf->grid->Nr; ++ir1) {
//		double const r1 = (ir1+1)*wf->grid->dr;
//		res += func(wf, ir1, il, r, r1);
//	}
//
//	return res*wf->grid->dr;
//}
//
//typedef double (func_t)(int ir);
///*!\fn
// * \brief \f[F_l(r, f) = \int dr' \frac{r_<^l}{r_>^{l+1}} f(r')\f]
// * For fast calculation split integer into two parts: \f[int = \int_0^r + \int_r\f]
// * */
//double F(int l, int ir, func_t f, sphere_grid_t const* grid) {
//	return F_down(l, ir, f, grid) + F_up(l, ir, f, grid);
//}
//
///*!\fn
// * \brief \f[F_l(r, f) = \int_0^r dr' \frac{r_<^l}{r_>^{l+1}} f(r')\f]
// * */
//double F_down(int l, int ir, func_t f, sphere_grid_t const* grid) {
//	double res = 0.0;
//	double const r = (ir+1)*grid->dr;
//
//	// int_r' then r' < r
//	for (int ir1 = 0; ir1 < ir; ++ir1) {
//		double const r1 = (ir1+1)*grid->dr;
//		res += func(ir1) * pow(r1, l) / pow(r, l+1);
//	}
//
//	return res*grid->dr;
//}
//
//double F_down_next(double F_down, int l, int ir, func_t f, sphere_grid_t const* grid) {
//	return F_down*pow()/pow();
//}
//
///*!\fn
// * \brief \f[F_l(r, f) = \int_r dr' \frac{r_<^l}{r_>^{l+1}} f(r')\f]
// * */
//double F_up(int l, int ir, func_t f, sphere_grid_t const* grid) {
//	double res = 0.0;
//	double const r = (ir+1)*grid->dr;
//
//	// int_r' then r' > r
//	for (int ir1 = ir; ir1 < wf->grid->Nr; ++ir1) {
//		double const r1 = (ir1+1)*wf->grid->dr;
//		res += func(ir1) * pow(r, l) / pow(r1, l+1);
//	}
//
//	return res*grid->dr;
//}
//
///*!\fn
// * \brief \f[F_l(r, f) = \int dr' \frac{r_<^l}{r_>^{l+1}} f(r')\f]
// * */
//double F_next(double F_prev_down, double F_prev_up, int l, int ir, func_t f, sphere_grid_t const* grid) {
//	double res = 0.0;
//	double const r = (ir+1)*grid->dr;
//
//	// int_r' then r' < r
//	for (int ir1 = 0; ir1 < ir; ++ir1) {
//		double const r1 = (ir1+1)*grid->dr;
//		res += func(ir1) * pow(r1, l) / pow(r, l+1);
//	}
//
//	// int_r' then r' > r
//	for (int ir1 = ir; ir1 < wf->grid->Nr; ++ir1) {
//		double const r1 = (ir1+1)*wf->grid->dr;
//		res += func(ir1) * pow(r, l) / pow(r1, l+1);
//	}
//
//	return res*grid->dr;
//}
//
//void hartree_potential_l0(int Ne, sphere_wavefunc_t const wf[Ne], double U[wf[0].grid->Nr]) {
//	sphere_grid_t const* grid = wf[0].grid;
//
//	// r
//	for (int ir = 0; ir < grid->Nr; ++ir) {
//		U[ir] = 0.0;
//		// sum_i
//		for (int ie = 0; ie < Ne; ++ie) {
//			// sum_l
//			for (int il = 0; il < grid->Nl; ++il) {
//				U[ir] += _integrate(&wf[ie], ir, il, _l0);
//			}
//		}
//		U[ir] *= 2; // spin
//	}
//}
//
//inline double _l1(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax) {
//	return swf_get(wf, ir, il) * (
//			clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1)) +
//			clm(il  , wf->m)*conj(swf_get(wf, ir, il+1))
//			) * rmin / pow(rmax, 2);
//}
//
//inline double _l1_0(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax) {
//	il = 0;
//	return swf_get(wf, ir, il) * (
//			clm(il  , wf->m)*conj(swf_get(wf, ir, il+1))
//			) * rmin / pow(rmax, 2);
//}
//
//inline double _l1_L(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax) {
//	il = wf->grid->Nl;
//	return swf_get(wf, ir, il) * (
//			clm(il-1, wf->m)*conj(swf_get(wf, ir, il-1))
//			) * rmin / pow(rmax, 2);
//}
//
//void hartree_potential_l1(int Ne, sphere_wavefunc_t const wf[Ne], cdouble U[wf[0].grid->Nr]) {
//	sphere_grid_t const* grid = wf[0].grid;
//	// r
//	for (int ir = 0; ir < grid->Nr; ++ir) {
//		U[ir] = 0.0;
//		// sum_i
//		for (int ie = 0; ie < Ne; ++ie) {
//			// sum_l
//			{
//				int const il = 0;
//				U[ir] += _integrate(&wf[ie], ir, il, _l1_0);
//			}
//
//			for (int il = 1; il < grid->Nl-1; ++il) {
//				U[ir] += _integrate(&wf[ie], ir, il, _l1);
//			}
//
//			{
//				int const il = grid->Nl-1;
//				U[ir] += _integrate(&wf[ie], ir, il, _l1_L);
//			}
//
//		}
//		U[ir] *= 2; // spin
//	}
//}
//
//inline double _l2_01(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax) {
//	return swf_get(wf, ir, il) * (
//			plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
//			qlm(il, wf->m)*conj(swf_get(wf, ir, il+2))
//			) * pow(rmin, 2) / pow(rmax, 3);
//}
//
//inline double _l2(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax) {
//	return swf_get(wf, ir, il) * (
//			plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
//			qlm(il, wf->m)*conj(swf_get(wf, ir, il+2)) +
//			qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
//			) * pow(rmin, 2) / pow(rmax, 3);
//}
//
//inline double _l2_L(sphere_wavefunc_t const* wf, int ir, int il, double rmin, double rmax) {
//	return swf_get(wf, ir, il) * (
//			plm(il, wf->m)*conj(swf_get(wf, ir, il)) +
//			qlm(il-2, wf->m)*conj(swf_get(wf, ir, il-2))
//			) * pow(rmin, 2) / pow(rmax, 3);
//}
//
//void hartree_potential_l2(int Ne, sphere_wavefunc_t const wf[Ne], cdouble U[wf[0].grid->Nr]) {
//	sphere_grid_t const* grid = wf[0].grid;
//	// r
//	for (int ir = 0; ir < grid->Nr; ++ir) {
//		U[ir] = 0.0;
//		// sum_i
//		for (int ie = 0; ie < Ne; ++ie) {
//			// sum_l
//			for (int il = 0; il < 2; ++il) {
//				U[ir] += _integrate(&wf[ie], ir, il, _l2_01);
//			}
//
//			for (int il = 2; il < grid->Nl-2; ++il) {
//				U[ir] += _integrate(&wf[ie], ir, il, _l2);
//			}
//
//			for (int il = grid->Nl-2; il < grid->Nl; ++il) {
//				U[ir] += _integrate(&wf[ie], ir, il, _l2_L);
//			}
//		}
//		U[ir] *= 2; // spin
//	}
//}

void ux_lda(int l, ks_orbitals_t const* orbs, double U[orbs->wf[0]->grid->n[iR]], sp_grid_t const* sp_grid) {
	double func(int ir, int ic) {
		return pow(ks_orbitals_n(orbs, (int[2]){ir, ic}), 1.0/3.0);
	}

	sh_series(func, l, 0, sp_grid, U);
}
