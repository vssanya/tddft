#include "atom.h"
#include "orbitals.h"

#include <array>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif


const std::vector<Atom::State> HAtom::GroundStateOrbs = {
	State("1s")
}; 

const std::vector<Atom::State> HeAtom::GroundStateOrbs = {
	State("1s")
}; 

const std::vector<Atom::State> MgAtom::GroundStateOrbs = {
    State("1s"), State("2s"), State("3s"),
    State("2p"), State("2p", 1, 4)
};

const std::vector<Atom::State> NaAtom::GroundStateOrbs = {
	State("1s", 0, 1, -1), State("2s", 0, 1, -1), State("3s",  0, 1, -1),
	State("2p", 0, 1, -1), State("2p", 1, 1, -1), State("2p", -1, 1, -1),
	State("1s", 0, 1,  1), State("2s", 0, 1,  1),
	State("2p", 0, 1,  1), State("2p", 1, 1,  1), State("2p",  1, 1,  1)
};

const std::vector<Atom::State> NeAtom::GroundStateOrbs = {
	State("1s"), State("2s"), // m = 0
	State("2p"),              // m = 0
	State("2p", 1, 4)         // m = +- 1
};

const std::vector<Atom::State> ArAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), // m = 0
	State("2p"), State("3p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4)   // m = +- 1
};

const std::vector<Atom::State> KrAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), // m = 0
	State("2p"), State("3p"), State("4p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4), State("4p", 1, 4), // m = +- 1
	State("3d"), State("3d", 1, 4), State("3d", 2, 4)
};

const std::vector<Atom::State> XeAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), State("5s"), // m = 0
	State("2p"), State("3p"), State("4p"), State("5p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4), State("4p", 1, 4), State("5p", 1, 4), // m = +- 1
	State("3d"), State("4d"),
	State("3d", 1, 4), State("4d", 1, 4),
	State("3d", 2, 4), State("4d", 2, 4)
};

const std::array<double, 2*3> Na_B {{
    6.46644991, 2.03040457,
    9.07195947, 1.22049052,
    3.66561470, 3.88900584
}};

const std::array<double, 2*3> Na_C {{
    0.35071677, 0.64928323,
    1.00486813, -0.05093639,
    1.06629058, 0.70089565,
}};

/*
extern constexpr std::array<double, 2*5> rb_B {{
	7.83077875,  2.75163799,
	4.30010258,  0.0,
	43.31975597, 0.0,
	2.93818679,  0.0,
	4.97097146,  0.0
}};

extern constexpr std::array<double, 2*5> rb_C {{
	0.81691787,   0.18308213,
	2.53670563,   0.0,
	-19.56508990, 0.0,
	1.06320272,   0.0,
	-0.99934358,  0.0
}};

double atom_u_rb_sae(Atom const* atom, ShGrid const* grid, int ir) {
	return potential_sgb_u<37, 37, 5, 2, rb_C, rb_B>(grid->r(ir));
}

double atom_dudz_rb_sae(Atom const* atom, ShGrid const* grid, int ir) {
	return potential_sgb_dudz<37, 37, 5, 2, rb_C, rb_B>(grid->r(ir));
}
*/


void atom_hydrogen_ground(ShWavefunc* wf) {
	assert(wf->m == 0);
	// l = 0
	{
		int const il = 0;
		for (int ir = 0; ir < wf->grid.n[iR]; ++ir) {
            double r = wf->grid.r(ir);
			(*wf)(ir, il) = 2*r*exp(-r);
		}
	}

	for (int il = 1; il < wf->grid.n[iL]; ++il) {
		for (int ir = 0; ir < wf->grid.n[iR]; ++ir) {
			(*wf)(ir, il) = 0.0;
		}
	}
}


template <typename Grid>
AtomCache<Grid>::AtomCache(Atom const& atom, Grid const& grid, double* u): 
	atom(atom),
	grid(grid),
	gpu_data_u(nullptr),
	gpu_data_dudz(nullptr)
{
	const int Nr = grid.n[iR];
	data_u = new double[Nr];
	data_dudz = new double[Nr];

	if (u == nullptr) {
#pragma omp parallel for
		for (int ir=0; ir<Nr; ir++) {
			double r = grid.r(ir);
			data_u[ir] = atom.u(r);
			data_dudz[ir] = atom.dudz(r);
		}
	} else {
#pragma omp parallel for
		for (int ir=0; ir<Nr; ir++) {
			data_u[ir] = u[ir] + atom.u(grid.r(ir));
		}

#pragma omp parallel for
		for (int ir=0; ir<Nr; ir++) {
			data_dudz[ir] = grid.d_dr(u, ir) + atom.dudz(grid.r(ir));
		}
	}
}

template <typename Grid>
AtomCache<Grid>::~AtomCache() {
	delete[] data_u;
	delete[] data_dudz;

#ifdef WITH_CUDA
	if (gpu_data_u != nullptr) {
		cudaFree(gpu_data_u);
	}

	if (gpu_data_dudz != nullptr) {
		cudaFree(gpu_data_dudz);
	}
#endif
}

template <typename Grid>
double* AtomCache<Grid>::getGPUDataU() {
#ifdef WITH_CUDA
	if (gpu_data_u == nullptr) {
		auto size = sizeof(double)*grid.n[iR];
		cudaMalloc(&gpu_data_u, size);
		cudaMemcpy(gpu_data_u, data_u, size, cudaMemcpyHostToDevice);
	}

	return gpu_data_u;
#else
	assert(false);
#endif
}

template <typename Grid>
double* AtomCache<Grid>::getGPUDatadUdz() {
#ifdef WITH_CUDA
	if (gpu_data_dudz == nullptr) {
		auto size = sizeof(double)*grid.n[iR];
		cudaMalloc(&gpu_data_dudz, size);
		cudaMemcpy(gpu_data_dudz, data_dudz, size, cudaMemcpyHostToDevice);
	}

	return gpu_data_dudz;
#else
	assert(false);
#endif
}

template class AtomCache<ShGrid>;
template class AtomCache<ShNotEqudistantGrid>;
