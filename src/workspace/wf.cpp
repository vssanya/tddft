#include <stdlib.h>
#include <stdio.h>

#include "wf.h"
#include "common_alg.h"


template <typename Grid>
workspace::WavefuncWS<Grid>::WavefuncWS(
		Grid    const& grid,
		AtomCache<Grid> const* atom_cache,
		UabsCache const& uabs,
		PropAtType propAtType,
		Gauge gauge,
		int num_threads
		):
	grid(grid),
	atom_cache(atom_cache),
	uabs(uabs),
	propAtType(propAtType),
	gauge(gauge),
	num_threads(num_threads)
{
#ifdef _OPENMP
	int max_threads = omp_get_max_threads();
	if (num_threads < 1 || num_threads > max_threads) {
		num_threads = max_threads;
	}
#else
	num_threads = 1;
#endif

	alpha = new cdouble[grid.n[iR]*num_threads]();
	betta = new cdouble[grid.n[iR]*num_threads]();
}

template <typename Grid>
void workspace::WavefuncWS<Grid>::prop_at_Odr4(Wavefunc<Grid>& wf, cdouble dt, sh_f Ul) {
	int const Nr = grid.n[iR];
	double const dr = grid.d[iR];
	double const dr2 = dr*dr;

	int const Z = atom_cache->atom.Z;


	double const d2[3] = {1.0/dr2, -2.0/dr2, 1.0/dr2};
	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	double const M2[3] = {
		1.0/12.0,
		10.0/12.0,
		1.0/12.0
	};

	const double M2_l0_11 = (1.0 + d2_l0_11*dr2/12.0);

	double U[3];
	cdouble al[3];
	cdouble ar[3];
	cdouble f;

#pragma omp for private(U, al, ar, f)
	for (int l = wf.m; l < wf.grid.n[iL]; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha_tid = &alpha[tid*Nr];
		cdouble* betta_tid = &betta[tid*Nr];

		cdouble* psi = &wf(0,l);

		cdouble const idt_2 = 0.5*I*dt;

		{
			int ir = 0;

			U[1] = Ul(ir  , l, wf.m);
			U[2] = Ul(ir+1, l, wf.m);

			for (int i = 1; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			if (l == 0 && atom_cache->atom.potentialType == Atom::POTENTIAL_COULOMB) {
				al[1] = M2_l0_11*(1.0 + idt_2*U[1]) - 0.5*idt_2*d2_l0_11;
				ar[1] = M2_l0_11*(1.0 - idt_2*U[1]) + 0.5*idt_2*d2_l0_11;
			}

			f = ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[0] = -al[2]/al[1];
			betta_tid[0] = f/al[1];
		}

		for (int ir = 1; ir < Nr-1; ++ir) {
			U[0] = U[1];
			U[1] = U[2];
			U[2] = Ul(ir+1, l, wf.m);

			for (int i = 0; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[ir] = - al[2] / c;
			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		{
			int ir = Nr-1;

			U[0] = U[1];
			U[1] = U[2];

			for (int i = 0; i < 2; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir];

			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		psi[Nr-1] = betta_tid[Nr-1];
		for (int ir = Nr-2; ir >= 0; --ir) {
			psi[ir] = alpha_tid[ir]*psi[ir+1] + betta_tid[ir];
		}
	}

}

template <typename Grid>
void workspace::WavefuncWS<Grid>::prop_at_Odr3(Wavefunc<Grid>& wf, cdouble dt, sh_f Ul) {
	int const Nr = grid.n[iR];

	cdouble al[3];
	cdouble ar[3];
	cdouble f;

#pragma omp for private(al, ar, f)
	for (int l = wf.m; l < wf.grid.n[iL]; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha_tid = &alpha[tid*Nr];
		cdouble* betta_tid = &betta[tid*Nr];

		cdouble* psi = &wf(0,l);

		cdouble const idt_2 = 0.5*I*dt;

		{
			int ir = 0;

			double U = Ul(ir, l, wf.m);

			for (int i = 1; i < 3; ++i) {
				al[i] = - 0.5*idt_2*grid.d2(ir, i);
				ar[i] = + 0.5*idt_2*grid.d2(ir, i);
			}

			if (false) {//l == 0 && atom_cache.atom.potentialType == Atom::POTENTIAL_COULOMB) {
				//al[1] = 1.0 + idt_2*U - 0.5*idt_2*d2_l0_11;
				//ar[1] = 1.0 - idt_2*U + 0.5*idt_2*d2_l0_11;
			} else {
				al[1] = 1.0 + idt_2*U - 0.5*idt_2*grid.d2(ir, 1);
				ar[1] = 1.0 - idt_2*U + 0.5*idt_2*grid.d2(ir, 1);
			}

			f = ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[0] = -al[2]/al[1];
			betta_tid[0] = f/al[1];
		}

		for (int ir = 1; ir < Nr-1; ++ir) {
			double U = Ul(ir, l, wf.m);

			for (int i = 0; i < 3; ++i) {
				al[i] = - 0.5*idt_2*grid.d2(ir, i);
				ar[i] = + 0.5*idt_2*grid.d2(ir, i);
			}

			al[1] += 1.0 + idt_2*U;
			ar[1] += 1.0 - idt_2*U;

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[ir] = - al[2] / c;
			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		{
			int ir = Nr-1;

			double U = Ul(ir, l, wf.m);

			for (int i = 0; i < 2; ++i) {
				al[i] = - 0.5*idt_2*grid.d2(ir, i);
				ar[i] = + 0.5*idt_2*grid.d2(ir, i);
			}

			al[1] += 1.0 + idt_2*U;
			ar[1] += 1.0 - idt_2*U;

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir];

			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		psi[Nr-1] = betta_tid[Nr-1];
		for (int ir = Nr-2; ir >= 0; --ir) {
			psi[ir] = alpha_tid[ir]*psi[ir+1] + betta_tid[ir];
		}
	}
}


template <typename Grid>
void workspace::WavefuncWS<Grid>::prop_mix(Wavefunc<Grid>& wf, sh_f Al, double dt, int l) {
	int    const Nr = grid.n[iR];
	double const dr = grid.d[iR];

	cdouble* v[2] = {&wf(0,l), &wf(0,l+1)};
	linalg::matrix_dot_vec(Nr, v, linalg::matrix_bE::dot);

	double const glm = -dt*Al(0, l, wf.m)/(4.0*dr);
	const double x = sqrt(3.0) - 2.0;

	linalg::tdm_t M = {(4.0+x)/6.0, (4.0+x)/6.0, {1.0/6.0, 2.0/3.0, 1.0/6.0}, Nr};

#pragma omp single nowait
	{
		int tid = omp_get_thread_num();
		linalg::eq_solve(v[0], M, {-x*glm,  x*glm, { glm, 0.0, -glm}, Nr}, &alpha[tid*Nr], &betta[tid*Nr]);
	}
#pragma omp single
	{
		int tid = omp_get_thread_num();
		linalg::eq_solve(v[1], M, { x*glm, -x*glm, {-glm, 0.0,  glm}, Nr}, &alpha[tid*Nr], &betta[tid*Nr]);
	}

	linalg::matrix_dot_vec(Nr, v, linalg::matrix_bE::dot_T);
}

template <typename Grid>
void workspace::WavefuncWS<Grid>::prop_abs(Wavefunc<Grid>& wf, double dt) {
	assert(wf.grid.n[iR] == grid.n[iR]);
	assert(wf.grid.n[iL] <= grid.n[iL]);
#pragma omp parallel for collapse(2)
	for (int il = wf.m; il < wf.grid.n[iL]; ++il) {
		for (int ir = 0; ir < wf.grid.n[iR]; ++ir) {
			wf(ir, il) *= exp(-uabs.u(ir)*dt);
		}
	}
}

template <typename Grid>
void workspace::WavefuncWS<Grid>::prop_common(Wavefunc<Grid>& wf, cdouble dt, int l_max, sh_f* Ul, sh_f* Al) {
	assert(wf.grid.n[iR] == grid.n[iR]);
	assert(wf.grid.n[iL] <= grid.n[iL]);
	const int Nl = wf.grid.n[iL];
#pragma omp parallel
	{
		for (int l1 = 1; l1 < l_max; ++l1) {
			for (int il = wf.m; il < Nl - l1; ++il) {
				wf_prop_ang_E_l(wf, 0.5*dt, il, l1, Ul[l1]);
			}
		}

		if (Al != nullptr) {
			for (int il=wf.m; il<Nl-1; ++il) {
				wf_prop_ang_A_l(wf, dt*0.5, il, 1, Al[1]);
			}

			for (int il=wf.m; il<Nl-1; ++il) {
				prop_mix(wf, Al[0], creal(dt*0.5), il);
			}
		}

		prop_at(wf, dt, Ul[0]);

		if (Al != nullptr) {
			for (int il=Nl-2; il>=wf.m; --il) {
				prop_mix(wf, Al[0], creal(dt*0.5), il);
			}

			for (int il=Nl-2; il>=wf.m; --il) {
				wf_prop_ang_A_l(wf, dt*0.5, il, 1, Al[1]);
			}
		}

		for (int l1 = l_max-1; l1 > 0; --l1) {
			for (int il = Nl - 1 - l1; il >= wf.m; --il) {
				wf_prop_ang_E_l(wf, 0.5*dt, il, l1, Ul[l1]);
			}
		}

	}
}

template <typename Grid>
void workspace::WavefuncWS<Grid>::prop(Wavefunc<Grid>& wf, double Et, double At, double dt) {
	sh_f Ul[2] = {
		[this](int ir, int l, int m) -> double {
			double const r = grid.r(ir);
			return l*(l+1)/(2*r*r) + atom_cache->u(ir);
		},
		[this, Et](int ir, int l, int m) -> double {
			double const r = grid.r(ir);
			return r*Et*clm(l,m);
		}
	};

	sh_f Al[2] = {
		[At](int ir, int l, int m) -> double {
			return At*clm(l,m);
		},
		[this, At](int ir, int l, int m) -> double {
			double const r = grid.r(ir);
			return At*(l+1)*clm(l,m)/r;
		}
	};

	switch (gauge) {
		case Gauge::LENGTH:
			prop_common(wf, dt, 2, Ul);
			break;
		case Gauge::VELOCITY:
			prop_common(wf, dt, 1, Ul, Al);
			break;
	}

	prop_abs(wf, dt);
}

template <typename Grid>
void workspace::WavefuncWS<Grid>::prop_without_field(Wavefunc<Grid>& wf, double dt) {
	sh_f Ul[1] = {
		[this](int ir, int l, int m) -> double {
			double const r = grid.r(ir);
			return l*(l+1)/(2*r*r) + atom_cache->u(ir);
		},
	};

	prop_common(wf, dt, 1, Ul);
	prop_abs(wf, dt);
}

template <typename Grid>
void workspace::WavefuncWS<Grid>::prop_img(Wavefunc<Grid>& wf, double dt) {
	sh_f Ul[1] = {
		[this](int ir, int l, int m) -> double {
			double const r = grid.r(ir);
			return l*(l+1)/(2*r*r) + atom_cache->u(ir);
		}
	};

	prop_common(wf, -I*dt, 1, Ul);
}

template<>
void workspace::WavefuncWS<ShNotEqudistantGrid>::prop_at_Odr4(Wavefunc<ShNotEqudistantGrid>& wf, cdouble dt, sh_f Ul) {
	int const Nr = grid.n[iR];

	int const Z = atom_cache->atom.Z;


	double dr1 = grid.dr(0);
	double dr2 = grid.dr(1);

	double dr1_dr2 = dr1*(3*dr1 + 4*dr2)*Z - 6*(dr1 + dr2);
	double d2_l0_11 = 2*(dr1+dr2)*(6*dr1 - (3*dr1 - dr2)*(dr1+dr2)*Z)/(dr1*dr1*dr2*dr1_dr2);
	double d2_l_11  = - 2*(3*dr1-dr2)*(dr1+dr2)*(dr1+dr2)/(dr1*dr1*dr1*dr2*(3*dr1 + 4*dr2));

	double M2_l0_11 = (dr1+dr2)*(dr1*(dr1+dr2)*(dr1+3*dr2)*Z - 3*(dr1*dr1 + 3*dr2*dr1 + dr2*dr2))/(3*dr1*dr2*dr1_dr2);
	double M2_l0_12 = (-dr1*dr1*dr1*Z + dr1*dr1*(dr2*Z + 3) + dr1*dr2*(2*dr2*Z - 3) - 3*dr2*dr2)/(3*dr2*dr1_dr2);

	double M2_l_11 = (dr1+dr2)*(dr1+dr2)*(dr1+3*dr2)/(3*dr1*dr2*(3*dr1+4*dr2));
	double M2_l_12 = -(dr1-2*dr2)*(dr1+dr2)/(3*dr2*(3*dr1+4*dr2));

	double U[3];
	cdouble al[3];
	cdouble ar[3];
	cdouble f;

	std::function<double(double, double)> M2[3] = {
		[](double d1, double d2) -> double {
			return (d1*d1 + d1*d2 - d2*d2)/(6*d1*(d1+d2));
		},
		[](double d1, double d2) -> double {
			return (d1*d1 + 3*d1*d2 + d2*d2)/(6*d1*d2);
		},
		[](double d1, double d2) -> double {
			return (-d1*d1 + d1*d2 + d2*d2)/(6*d2*(d1+d2));
		}
	};

	std::function<double(double, double)> d2[3] = {
		[](double d1, double d2) -> double {
			return 2.0/(d1*(d1+d2));
		},
		[](double d1, double d2) -> double {
			return -2.0/(d1*d2);
		},
		[](double d1, double d2) -> double {
			return 2.0/(d2*(d1+d2));
		}
	};

#pragma omp for private(U, al, ar, f)
	for (int l = wf.m; l < wf.grid.n[iL]; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha_tid = &alpha[tid*Nr];
		cdouble* betta_tid = &betta[tid*Nr];

		cdouble* psi = &wf(0,l);

		cdouble const idt_2 = 0.5*I*dt;

		{
			int ir = 0;

			U[1] = Ul(ir  , l, wf.m);
			U[2] = Ul(ir+1, l, wf.m);

			double dr1 = grid.dr(ir);
			double dr2 = grid.dr(ir+1);

			if (l == 0 && atom_cache->atom.potentialType == Atom::POTENTIAL_COULOMB) {
				int i = 1;
				al[i] = M2_l0_11*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2_l0_11;
				ar[i] = M2_l0_11*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2_l0_11;

				i = 2;
				al[i] = M2_l0_12*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
				ar[i] = M2_l0_12*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
			} else {
				int i = 1;
				al[i] = M2_l_11*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2_l_11;
				ar[i] = M2_l_11*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2_l_11;

				i = 2;
				al[i] = M2_l_12*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
				ar[i] = M2_l_12*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
			}

			f = ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[0] = -al[2]/al[1];
			betta_tid[0] = f/al[1];
		}

		for (int ir = 1; ir < Nr-1; ++ir) {
			U[0] = U[1];
			U[1] = U[2];
			U[2] = Ul(ir+1, l, wf.m);

			double dr1 = grid.dr(ir);
			double dr2 = grid.dr(ir+1);

			for (int i = 0; i < 3; ++i) {
				al[i] = M2[i](dr1, dr2)*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
				ar[i] = M2[i](dr1, dr2)*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
			}

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir] + ar[2]*psi[ir+1];

			alpha_tid[ir] = - al[2] / c;
			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		{
			int ir = Nr-1;

			U[0] = U[1];
			U[1] = U[2];

			double dr1 = grid.dr(ir-1);
			double dr2 = grid.dr(ir);

			for (int i = 0; i < 2; ++i) {
				al[i] = M2[i](dr1, dr2)*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i](dr1, dr2);
				ar[i] = M2[i](dr1, dr2)*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i](dr1, dr2);
			}

			cdouble c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir];

			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		psi[Nr-1] = betta_tid[Nr-1];
		for (int ir = Nr-2; ir >= 0; --ir) {
			psi[ir] = alpha_tid[ir]*psi[ir+1] + betta_tid[ir];
		}
	}
}

template class workspace::WavefuncWS<ShGrid>;
template class workspace::WavefuncWS<ShNotEqudistantGrid>;


template <typename Grid>
workspace::Wavefunc3DWS<Grid>::Wavefunc3DWS(
		Grid    const& grid,
		AtomCache<Grid> const& atom_cache,
		UabsCache const& uabs,
		PropAtType propAtType,
		int num_threads
		):
	grid(grid),
	atom_cache(atom_cache),
	uabs(uabs),
	propAtType(propAtType),
	num_threads(num_threads)
{
#ifdef _OPENMP
	int max_threads = omp_get_max_threads();
	if (num_threads < 1 || num_threads > max_threads) {
		num_threads = max_threads;
	}
#else
	num_threads = 1;
#endif

	alpha = new Array2D<cdouble>(Grid2d(grid.n[iR], num_threads));
	betta = new Array2D<cdouble>(Grid2d(grid.n[iR], num_threads));
}

template <typename Grid>
void workspace::Wavefunc3DWS<Grid>::prop_abs(ShWavefunc3D<Grid>& wf, double dt) {
	assert(wf.grid.n[iR] == grid.n[iR]);
	assert(wf.grid.n[iL] <= grid.n[iL]);

#pragma omp parallel for
	for (int l=0; l<wf.grid.n[iL]; ++l) {
		for (int m=-l; m<=l; ++m) {
			for (int ir=0; ir<wf.grid.n[iR]; ++ir) {
				wf(ir, l, m) *= exp(-uabs.u(ir)*dt);
			}
		}
	}
}

template <typename Grid>
void workspace::Wavefunc3DWS<Grid>::prop_at(ShWavefunc3D<Grid>& wf, cdouble dt) {
	const int Nl = wf.grid.n[iL];

#pragma omp for
	for (int l = 0; l < Nl; ++l) {
		int tid = omp_get_thread_num();

		cdouble* alpha_tid = &(*alpha)(0, tid);
		cdouble* betta_tid = &(*betta)(0, tid);

		auto useBorderCondition = l == 0 && atom_cache.atom.potentialType == Atom::POTENTIAL_COULOMB;
		auto Ur = [l, this](int ir) -> double {
			double const r = grid.r(ir);
			return l*(l+1)/(2*r*r) + atom_cache.u(ir);
		};

		for (int m=-l; m<=l; ++m) {
			wf_prop_at_Odr4(wf.slice(l, m), dt, Ur,
					useBorderCondition, atom_cache.atom.Z,
					alpha_tid, betta_tid);
		}
	}
}

template <typename Grid>
void workspace::Wavefunc3DWS<Grid>::prop(
		ShWavefunc3D<Grid>& wf,
		field_t const* Fx,
		field_t const* Fy,
		double t, double dt
		) {
	assert(wf.grid.n[iR] == grid.n[iR]);
	assert(wf.grid.n[iL] <= grid.n[iL]);
	const int Nl = wf.grid.n[iL];

	cdouble E = field_E(Fx, t+dt*0.5) - I*field_E(Fy, t+dt*0.5);

	double phi = carg(E);
	double mod_E = cabs(E);

	auto Hang_1 = [this, mod_E](int ir, int l, int m) {
		return 0.5*grid.r(ir)*mod_E*std::sqrt((l-m-1)*(l-m)/(2*l-1)/(2*l+1));
	};

	auto Hang_2 = [this, mod_E](int ir, int l, int m) {
		return -0.5*grid.r(ir)*mod_E*std::sqrt((l+m-1)*(l+m)/(2*l-1)/(2*l+1));
	};

#pragma omp parallel
	{
		for (int l = 0; l < Nl-1; ++l) {
			for (int m = -l; m <= l; ++m) {
				if (m != -l)
					wf_prop_ang_E_lm(wf, 0.5*dt, l, 1, m, -1, phi, Hang_1);

				if (m != l)
					wf_prop_ang_E_lm(wf, 0.5*dt, l, 1, m, 1, phi, Hang_2);
			}
		}

		prop_at(wf, dt);

		for (int l = Nl - 2; l >= 0; --l) {
			for (int m = l; m >= -l; --m) {
				if (m != -l)
					wf_prop_ang_E_lm(wf, 0.5*dt, l, 1, m, -1, phi, Hang_1);

				if (m != l)
					wf_prop_ang_E_lm(wf, 0.5*dt, l, 1, m, 1, phi, Hang_2);
			}
		}
	}

	prop_abs(wf, dt);
}

template<typename Grid>
void workspace::Wavefunc3DWS<Grid>::prop_img(ShWavefunc3D<Grid>& wf, double dt) {
	prop_at(wf, -I*dt);
}

template class workspace::Wavefunc3DWS<ShNotEqudistantGrid3D>;
