#include "wf_gpu.h"

#include <cuda_runtime.h>
#include "../utils.h"


workspace::WfGpu::WfGpu(AtomCache const& atomCache, ShGrid const& grid, UabsCache const& uabsCache, int gpuGridNl, int threadsPerBlock):
	grid(grid),
	uabsCache(uabsCache),
	atomCache(atomCache),
	threadsPerBlock(threadsPerBlock)
{
    int Nr = grid.n[iR];

	d_gridNl = std::min(grid.n[iL], gpuGridNl);

    cudaMalloc(&d_alpha, sizeof(cdouble)*Nr*d_gridNl);
    cudaMalloc(&d_betta, sizeof(cdouble)*Nr*d_gridNl);

    cudaMalloc(&d_uabs, sizeof(double)*Nr);
    cudaMemcpy(d_uabs, uabsCache.data, sizeof(double)*Nr, cudaMemcpyHostToDevice);

    cudaMalloc(&d_atomU, sizeof(double)*Nr);
    cudaMemcpy(d_atomU, atomCache.data_u, sizeof(double)*Nr, cudaMemcpyHostToDevice);
}

workspace::WfGpu::~WfGpu() {
	cudaFree(d_alpha);
	cudaFree(d_betta);
	cudaFree(d_uabs);
	cudaFree(d_atomU);
}

__global__ void kernel_wf_prop_abs(cuComplex* wf, double* uabs, double dt, int Nr, int Nl) {
	int ir = blockIdx.x*blockDim.x + threadIdx.x;

	if (ir < Nr) {
		double u = exp(-uabs[ir]*dt);
		for (int il=0; il<Nl; il++) {
			wf[ir + il*Nr] *= u;
		}
	}
}

void workspace::WfGpu::prop_abs(ShWavefuncGPU& wf, double dt) {
	kernel_wf_prop_abs<<<div_up(grid.n[iR], threadsPerBlock), threadsPerBlock>>>((cuComplex*)wf.data, d_uabs, dt, grid.n[iR], grid.n[iL]);
}

// potentialType = 1 (POTENTIAL_COULOMB)
__global__ void kernel_wf_prop_at(cuComplex* wf, double* Ur, cuComplex* alpha, cuComplex* betta, int N, int Nr, int Nl, double dr, double dt, int Z, int potentialType) {
	int l0 = blockIdx.x*blockDim.x + threadIdx.x;

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

	cuComplex* alpha_tid = &alpha[l0*Nr];
	cuComplex* betta_tid = &betta[l0*Nr];

	for (int l = l0; l < Nl; l+=N) {
		cuComplex* psi = &wf[l*Nr];

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

			if (l == 0 && potentialType == 1) {
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

void workspace::WfGpu::prop_at(ShWavefuncGPU& wf, double dt) {
	kernel_wf_prop_at<<<d_gridNl/threadsPerBlock, threadsPerBlock>>>((cuComplex*) wf.data, d_atomU,  (cuComplex*) d_alpha,  (cuComplex*) d_betta, d_gridNl, grid.n[iR], grid.n[iL], grid.d[iR], dt, atomCache.atom.Z, atomCache.atom.potentialType == Atom::PotentialType::POTENTIAL_COULOMB ? 1 : 0);
}

__device__ void cu_wf_dot(cuComplex v[2]) {
	cuComplex res[2] = {
		v[0] + v[1],
		-v[0] + v[1]
	};

	v[0] = res[0];
	v[1] = res[1];
}

__device__ void cu_wf_dot_T(cuComplex v[2]) {
	cuComplex res[2] = {
		0.5*(v[0] - v[1]),
		0.5*(v[0] + v[1])
	};

	v[0] = res[0];
	v[1] = res[1];
}

__global__ void kernel_wf_prop_ang_l(cuComplex* wf, cuComplex dt, const cuComplex eigenval0, const cuComplex eigenval1, int m, int l, int l1, double E, double dr, int Nr, int Nl) {
	int ir = blockIdx.x*blockDim.x + threadIdx.x;

	cuComplex* psi_l0 = &wf[Nr*l];
	cuComplex* psi_l1 = &wf[Nr*(l+l1)];

	cuComplex i = cuComplex(0.0, 1.0);

	if (ir < Nr) {
		double const r = dr*(ir+1);
		double const U = r*E*clm(l, m);

		cuComplex x[2] = {psi_l0[ir], psi_l1[ir]};

		cu_wf_dot(x);
		x[0] *= exp(i*U*dt*eigenval0);
		x[1] *= exp(i*U*dt*eigenval1);
		cu_wf_dot_T(x);

		psi_l0[ir] = x[0];
		psi_l1[ir] = x[1];
	}
}

void workspace::WfGpu::prop(ShWavefuncGPU& wf, field_t const& field, double t, double dt) {
	double E = field_E(&field, t + dt/2);

	dim3 gridDim(div_up(grid.n[iR], threadsPerBlock));

	const int l_max = 2;
	const int Nl = grid.n[iL];

	cdouble eigenval0 = {-1.0, 0.0};
	cdouble eigenval1 = {1.0, 0.0};

	for (int l1 = 1; l1 < l_max; ++l1) {
		for (int il = 0; il < Nl - l1; ++il) {
			kernel_wf_prop_ang_l<<<gridDim, threadsPerBlock>>>((cuComplex*) wf.data, 0.5*dt, eigenval0, eigenval1, wf.m, il, l1, E, grid.d[iR], grid.n[iR], Nl);
		}
	}

	prop_at(wf, dt);

	for (int l1 = l_max-1; l1 > 0; --l1) {
		for (int il = Nl - 1 - l1; il >= 0; --il) {
			kernel_wf_prop_ang_l<<<gridDim, threadsPerBlock>>>((cuComplex*) wf.data, 0.5*dt, eigenval0, eigenval1, wf.m, il, l1, E, grid.d[iR], grid.n[iR], Nl);
		}
	}

	prop_abs(wf, dt);
}
