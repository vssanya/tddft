#include "atom.h"
#include "orbitals.h"

#include <array>
#include <iostream>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif


const std::vector<Atom::State> HAtom::GroundStateOrbs = {
	State("1s")
}; 

const std::vector<Atom::State> HeAtom::GroundStateOrbs = {
	State("1s")
}; 

const std::vector<Atom::State> LiAtom::GroundStateOrbs = {
	State("1s", 0, 1, -1),
	State("1s", 0, 1,  1),
	State("2s", 0, 1,  1),
}; 

const std::array<double, 2*3> He_B {{
	6.28379042, 0.23817624,
	3.60759648, 1.39690934,
	3.64179068, 0.81080622,
}};

const std::array<double, 2*3> He_C {{
	0.33740902, 0.66259098,
	1.67240025, -1.15584925,
	-1.10884184, -0.33200330
}};

const std::array<double, 2*3> Li_B {{
	4.24222625, 0.28528662,
	3.02021466, 2.10774534,
	8.79311865, 1.03714279
}};

const std::array<double, 2*3> Li_C {{
	0.27231840, 0.72768160,
	0.80062418, -1.12966971,
	-2.94600827, -0.84110551
}};

const std::array<double, 2*3> Ne_B {{
	4.68014471, 2.41322960,
	5.80903874, 2.90207510,
	4.51696279, 3.06518063
}};

const std::array<double, 2*3> Ne_C {{
	0.46087879, 0.53912121,
	0.42068967, 0.47271993,
	-1.12569309, 1.29942636
}};

const std::array<double, 2*3> Ar_B {{
	1.69045518, 2.48113492,
	83.81780982, 0.57781500,
	14.79181994, 1.83863745
}};

const std::array<double, 2*3> Ar_C {{
	-0.14865117, 1.14865117,
	-2.01010923, -0.01998004,
	-20.01846287, 0.53596000
}};

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

const std::vector<Atom::State> CaAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), // m = 0
	State("2p"), State("3p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4)   // m = +- 1
};

const std::vector<Atom::State> KrAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), // m = 0
	State("2p"), State("3p"), State("4p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4), State("4p", 1, 4), // m = +- 1
	State("3d"), State("3d", 1, 4), State("3d", 2, 4)
};

const std::vector<Atom::State> RbAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), State("5s", 0, 1), // m = 0
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

const std::vector<Atom::State> CsAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), State("5s"), State("6s", 0, 1), // m = 0
	State("2p"), State("3p"), State("4p"), State("5p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4), State("4p", 1, 4), State("5p", 1, 4), // m = +- 1
	State("3d"), State("4d"),
	State("3d", 1, 4), State("4d", 1, 4),
	State("3d", 2, 4), State("4d", 2, 4)
};

const std::vector<Atom::State> BaAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), State("5s"), State("6s"), // m = 0
	State("2p"), State("3p"), State("4p"), State("5p"),              // m = 0
	State("2p", 1, 4), State("3p", 1, 4), State("4p", 1, 4), State("5p", 1, 4), // m = +- 1
	State("3d"), State("4d"),
	State("3d", 1, 4), State("4d", 1, 4),
	State("3d", 2, 4), State("4d", 2, 4)
};

const std::vector<Atom::State> BaPAtom::GroundStateOrbs = {
	State("1s"), State("2s"), State("3s"), State("4s"), State("5s"), State("6s", 0, 1), // m = 0
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

const std::array<double, 2*5> Rb_B {{
	7.83077875, 2.75163799,
	4.30010258, 0.0,
	43.31975597, 0.0,
	2.93818679, 0.0,
	4.97097146, 0.0
}};

const std::array<double, 2*5> Rb_C {{
	0.81691787, 0.18308213,
	2.53670563, 0.0,
	-19.56508990, 0.0,
	1.06320272, 0.0,
	-0.99934358, 0.0
}};

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
AtomCache<Grid>::AtomCache(Atom const& atom, Grid const& grid, double* u, int N): 
	atom(atom),
	grid(grid),
	gpu_data_u(nullptr),
	gpu_data_dudz(nullptr)
{
	const int Nr = grid.n[iR];
	data_u = new double[Nr];
	data_dudz = new double[Nr];

	if (N == -1) {
		N = Nr;
	}

	if (u == nullptr) {
#pragma omp parallel for
		for (int ir=0; ir<Nr; ir++) {
			double r = grid.r(ir);
			data_u[ir] = atom.u(r);
			data_dudz[ir] = atom.dudz(r);
		}
	} else {
		int Nlast = Nr;
		if (N < Nr) {
			Nlast = N;

			for (int i=N-1; i>= 0; i--) {
				if (abs(u[i]*grid.r(i) - u[i-1]*grid.r(i-1)) < 1e-4*u[i]*grid.r(i)) {
					Nlast = i;
					break;
				}
			}
		}

#pragma omp parallel for
		for (int ir=0; ir<Nlast; ir++) {
			double r = grid.r(ir);
			data_u[ir] = u[ir] + atom.u(r);
			data_dudz[ir] = grid.d_dr(u, ir) + atom.dudz(r);
		}

		double Zee = u[Nlast] * grid.r(Nlast);

#pragma omp parallel for
		for (int ir=Nlast; ir<Nr; ir++) {
			double r = grid.r(ir);
			data_u[ir] = atom.u(r) + Zee/r;
			data_dudz[ir] = atom.dudz(r) - Zee/(r*r);
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
