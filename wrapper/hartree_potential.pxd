from wavefunc cimport Wavefunc
from orbitals cimport Orbitals
from grid cimport cSpGrid
from sphere_harmonics cimport cYlmCache

cdef extern from "hartree_potential.h":
    ctypedef double (*potential_xc_f)(double n, double x)
    void hartree_potential[Grid](Orbitals[Grid]* orbs, int l, double* U, double* U_local, double* f, int order)
    void hartree_potential_calc_int_func[Grid](Orbitals[Grid]* orbs, int l, double* f);
    void hartree_potential_wf_l0[Grid](Wavefunc[Grid]* wf, double* U, double* f, int order)
    double mod_grad_n(cSpGrid* grid, double* n, int ir, int ic)
    double ux_lda_func(double n)
    double uc_lda_func(double n)
    double uxc_lb(double n, double x)
    double uxc_lda(double n, double x)
    double uxc_lda_x(double n, double x)
    void uxc_calc_l[Grid](
        potential_xc_f uxc,
        int l, Orbitals[Grid]* orbs,
        double* U,
        cSpGrid* grid,
        double* n, # for calc using mpi
        double* n_local, # for calc using mpi
        cYlmCache* ylm_cache
    )
    void uxc_calc_l0[Grid](
        potential_xc_f uxc,
        int l, Orbitals[Grid]* orbs,
        double* U,
        cSpGrid* grid,
        double* n, # for calc using mpi
        double* n_local, # for calc using mpi
        cYlmCache* ylm_cache
    )

cdef class Uxc:
    cdef potential_xc_f cdata
    cdef str name

    @staticmethod
    cdef Uxc from_c_func(potential_xc_f func, str name)
