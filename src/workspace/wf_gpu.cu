#include "wf_gpu.h"
#include "../pycuda-complex.hpp"

workspace::WfGPUBase::WfGPUBase(AtomCache const* atom_cache, ShGrid const* grid, uabs_sh_t const* uabs, int num_threads):
    atom_cache(atom_cache),
	grid(grid),
	uabs(uabs),
	num_threads(num_threads)
{
	cudaMalloc(&alpha, sizeof(cuComplex)*grid->n[iR]*grid->n[iL]);
	cudaMalloc(&betta, sizeof(cuComplex)*grid->n[iR]*grid->n[iL]);
}

workspace::WfGPUBase::~WfGPUBase() {
	cudaFree(alpha);
	cudaFree(betta);
}

/*void workspace::WfGPUBase::prop_mix(ShWavefuncGPU& wf, sh_f Al, double dt, int l) {*/
	/*int    const Nr = grid->n[iR];*/
	/*double const dr = grid->d[iR];*/

	/*cdouble* v[2] = {&wf(0,l), &wf(0,l+1)};*/
	/*linalg::matrix_dot_vec(Nr, v, linalg::matrix_bE::dot);*/

	/*double const glm = -dt*Al(grid, 0, l, wf.m)/(4.0*dr);*/
	/*const double x = sqrt(3.0) - 2.0;*/

	/*linalg::tdm_t M = {(4.0+x)/6.0, (4.0+x)/6.0, {1.0/6.0, 2.0/3.0, 1.0/6.0}, Nr};*/

/*#pragma omp single nowait*/
	/*{*/
		/*int tid = omp_get_thread_num();*/
		/*linalg::eq_solve(v[0], M, {-x*glm,  x*glm, { glm, 0.0, -glm}, Nr}, &alpha[tid*Nr], &betta[tid*Nr]);*/
	/*}*/
/*#pragma omp single*/
	/*{*/
		/*int tid = omp_get_thread_num();*/
		/*linalg::eq_solve(v[1], M, { x*glm, -x*glm, {-glm, 0.0,  glm}, Nr}, &alpha[tid*Nr], &betta[tid*Nr]);*/
	/*}*/

	/*linalg::matrix_dot_vec(Nr, v, linalg::matrix_bE::dot_T);*/
/*}*/

// O(dr^4)
/*
 * \brief Расчет функции \f[\psi(t+dt) = exp(-iH_{at}dt)\psi(t)\f]
 *
 * \f[H_{at} = -0.5\frac{d^2}{dr^2} + U(r, l)\f]
 * \f[exp(iAdt) = \frac{1 - iA}{1 + iA} + O(dt^3)\f]
 *
 * \param[in,out] wf
 *
 */
__global__ void kernel_prop_at(cuComplex* wf, double* Ur, cuComplex* alpha, cuComplex* betta, int Nr, int Nl, double dr, cuComplex dt, int Z) {
	double const dr2 = dr*dr;

	double const d2[3] = {1.0/dr2, -2.0/dr2, 1.0/dr2};
	double const d2_l0_11 = d2[1]*(1.0 - Z*dr/(12.0 - 10.0*Z*dr));

	double const M2[3] = {
		1.0/12.0,
		10.0/12.0,
		1.0/12.0
	};

	const double M2_l0_11 = (1.0 + d2_l0_11*dr2/12.0);

	double U[3];
	cuComplex al[3];
	cuComplex ar[3];
	cuComplex f;

	int l = blockIdx.x*blockDim.x + threadIdx.x;

	if (l < Nl) {
		cuComplex* psi = &wf[l*Nr];
		cuComplex* alpha_tid = &alpha[l*Nr];
		cuComplex* betta_tid = &betta[l*Nr];

		auto Ul = [dr, Ur, l](int ir) -> double {
			double const r = dr*(ir+1);
			return l*(l+1)/(2*r*r) + Ur[ir];
		};

		cuComplex const idt_2 = 0.5*dt*cuComplex(0.0, 1.0);

		{
			int ir = 0;

			U[1] = Ul(ir  );
			U[2] = Ul(ir+1);

			for (int i = 1; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

            if (l == 0) { // && atom_cache->atom.potentialType == Atom::POTENTIAL_COULOMB) {
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
			U[2] = Ul(ir+1);

			for (int i = 0; i < 3; ++i) {
				al[i] = M2[i]*(1.0 + idt_2*U[i]) - 0.5*idt_2*d2[i];
				ar[i] = M2[i]*(1.0 - idt_2*U[i]) + 0.5*idt_2*d2[i];
			}

			cuComplex c = al[1] + al[0]*alpha_tid[ir-1];
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

			cuComplex c = al[1] + al[0]*alpha_tid[ir-1];
			f = ar[0]*psi[ir-1] + ar[1]*psi[ir];

			betta_tid[ir] = (f - al[0]*betta_tid[ir-1]) / c;
		}

		psi[Nr-1] = betta_tid[Nr-1];
		for (int ir = Nr-2; ir >= 0; --ir) {
			psi[ir] = alpha_tid[ir]*psi[ir+1] + betta_tid[ir];
		}
	}
}

//__global__ void kernel_prop_at(cuComplex* wf, double* Ur, cuComplex* alpha, cuComplex* betta, int Nr, int Nl, double dr, cuComplex dt, int Z) {

void workspace::WfGPUBase::prop_at(ShWavefuncGPU& wf, cdouble dt, double* Ur) {
	/*dim3 blockDim(1);*/
	/*dim3 gridDim(wf.grid->n[iL]);*/

	kernel_prop_at<<<1,1>>>(
			wf.data,
			Ur,
			alpha,
			betta,
			wf.grid->n[iR], wf.grid->n[iL], wf.grid->d[iR], dt, atom_cache->atom.Z);
}

/*void workspace::WfGPUBase::prop_common(ShWavefuncGPU& wf, cdouble dt, int l_max, sh_f* Ul, sh_f* Al) {*/
	/*assert(wf.grid->n[iR] == grid->n[iR]);*/
	/*assert(wf.grid->n[iL] <= grid->n[iL]);*/
	/*const int Nl = wf.grid->n[iL];*/
/*#pragma omp parallel*/
	/*{*/
		/*for (int l1 = 1; l1 < l_max; ++l1) {*/
			/*for (int il = 0; il < Nl - l1; ++il) {*/
				/*wf_prop_ang_E_l(wf, 0.5*dt, il, l1, Ul[l1]);*/
			/*}*/
		/*}*/

		/*if (Al != nullptr) {*/
			/*for (int il=0; il<Nl-1; ++il) {*/
				/*wf_prop_ang_A_l(wf, dt*0.5, il, 1, Al[1]);*/
			/*}*/

			/*for (int il=0; il<Nl-1; ++il) {*/
				/*prop_mix(wf, Al[0], creal(dt*0.5), il);*/
			/*}*/
		/*}*/

        /*prop_at(wf, dt, Ul[0]);*/

		/*if (Al != nullptr) {*/
			/*for (int il=Nl-2; il>=0; --il) {*/
				/*prop_mix(wf, Al[0], creal(dt*0.5), il);*/
			/*}*/

			/*for (int il=Nl-2; il>=0; --il) {*/
				/*wf_prop_ang_A_l(wf, dt*0.5, il, 1, Al[1]);*/
			/*}*/
		/*}*/

		/*for (int l1 = l_max-1; l1 > 0; --l1) {*/
			/*for (int il = Nl - 1 - l1; il >= 0; --il) {*/
				/*wf_prop_ang_E_l(wf, 0.5*dt, il, l1, Ul[l1]);*/
			/*}*/
		/*}*/

	/*}*/
/*}*/

__global__ void kernel_prop_abs(cuComplex* wf, double* uabs, double dt, int Nr, int Nl) {
	int ir = blockIdx.x*blockDim.x + threadIdx.x;
	int l  = blockIdx.y*blockDim.y + threadIdx.y;

	if (ir < Nr && l < Nl) {
		wf[ir + l*Nr] *= exp(-uabs[ir]*dt);
	}
}

void workspace::WfGPUBase::prop_abs(ShWavefuncGPU& wf, double dt) {
	assert(wf.grid->n[iR] == grid->n[iR]);
	assert(wf.grid->n[iL] <= grid->n[iL]);

	dim3 blockDim(1,1);
	dim3 gridDim(wf.grid->n[iR], wf.grid->n[iL]);

	kernel_prop_abs<<<gridDim, blockDim>>>((cuComplex*)wf.data, uabs->data, dt, wf.grid->n[iR], wf.grid->n[iL]);
}

/*void workspace::WfGPUBase::prop(ShWavefuncGPU& wf, field_t const* field, double t, double dt) {*/
	/*double Et = field_E(field, t + dt/2);*/

	/*sh_f Ul[2] = {*/
            /*[this](ShGrid const* grid, int ir, int l, int m) -> double {*/
				/*double const r = grid->r(ir);*/
                /*return l*(l+1)/(2*r*r) + atom_cache->u(ir);*/
			/*},*/
            /*[Et](ShGrid const* grid, int ir, int l, int m) -> double {*/
				/*double const r = grid->r(ir);*/
				/*return r*Et*clm(l,m);*/
			/*}*/
	/*};*/


    /*prop_common(wf, dt, 2, Ul);*/

	/*prop_abs(wf, dt);*/
/*}*/

/*void workspace::WfGPUE::prop(ShWavefuncGPU& wf, field_t const* field, double t, double dt) {*/
	/*double Et = field_E(field, t + dt/2);*/

	/*sh_f Ul[2] = {*/
            /*[this](ShGrid const* grid, int ir, int l, int m) -> double {*/
				/*double const r = grid->r(ir);*/
                /*return l*(l+1)/(2*r*r) + atom_cache->u(ir);*/
			/*},*/
            /*[Et](ShGrid const* grid, int ir, int l, int m) -> double {*/
				/*double const r = grid->r(ir);*/
				/*return r*Et*clm(l,m);*/
			/*}*/
	/*};*/


    /*prop_common(wf, dt, 2, Ul);*/

	/*prop_abs(wf, dt);*/
/*}*/

/*void workspace::WfGPUBase::prop_without_field(ShWavefuncGPU &wf, double dt) {*/
    /*sh_f Ul[1] = {*/
            /*[this](ShGrid const* grid, int ir, int l, int m) -> double {*/
                /*double const r = grid->r(ir);*/
                /*return l*(l+1)/(2*r*r) + atom_cache->u(ir);*/
            /*},*/
    /*};*/

    /*prop_common(wf, dt, 1, Ul);*/
	/*prop_abs(wf, dt);*/
/*}*/

/*void workspace::WfGPUBase::prop_img(ShWavefuncGPU& wf, double dt) {*/
	/*sh_f Ul[1] = {*/
        /*[this](ShGrid const* grid, int ir, int l, int m) -> double {*/
			/*double const r = grid->r(ir);*/
            /*return l*(l+1)/(2*r*r) + atom_cache->u(ir);*/
		/*}*/
	/*};*/

    /*prop_common(wf, -I*dt, 1, Ul);*/
/*}*/
