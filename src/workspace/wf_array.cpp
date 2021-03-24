#include "wf_array.h"

template <typename Grid>
workspace::WfArray<Grid>::WfArray(
		Grid    const& grid,
		AtomCache<Grid> const* atom_cache,
		UabsCache const& uabs,
		PropAtType propAtType,
		Gauge gauge,
		int num_threads
		): ws(grid, atom_cache, uabs, propAtType, gauge, num_threads) {}

template <typename Grid>
void workspace::WfArray<Grid>::prop(WavefuncArray<Grid>* arr, double E[], double dt) {
	double* Elocal;
	if (arr->is_root()) {
		Elocal = E;
	} else {
		Elocal = new double[arr->N]();
	}

#ifdef _MPI
	MPI_Bcast(Elocal, arr->N, MPI_DOUBLE, 0, arr->mpi_comm);
#endif

	for (int ie=0; ie<arr->N; ++ie) {
		if (arr->wf[ie] != nullptr) {
			ws.prop(arr->wf[ie][0], Elocal[ie], 0, dt);
		}
	}
}

template class workspace::WfArray<ShGrid>;
template class workspace::WfArray<ShNotEqudistantGrid>;
