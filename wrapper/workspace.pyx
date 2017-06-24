import numpy as np
cimport numpy as np

from atom cimport Atom
from grid cimport ShGrid, SpGrid
from wavefunc cimport SWavefunc
from field cimport Field
from orbitals cimport SOrbitals
from sphere_harmonics cimport YlmCache
from hartree_potential cimport Uxc
from hartree_potential import UXC_LB


cdef class Eigen:
    def __cinit__(self, ShGrid grid):
        self.cdata = eigen_ws_alloc(grid.data)

    def __dealloc__(self):
        if self.cdata != NULL:
            eigen_ws_free(self.cdata)

    def calc(self, Atom atom):
        eigen_calc_for_atom(self.cdata, atom.cdata)

    def get_eval(self):
        cdef double[:, ::1] array = <double[:self.cdata.grid.n[1],:self.cdata.grid.n[0]]>self.cdata.eval
        return np.asarray(array)

    def get_evec(self):
        cdef double[:, :, ::1] array = <double[:self.cdata.grid.n[1], :self.cdata.grid.n[0], :self.cdata.grid.n[0]]>self.cdata.evec
        return np.asarray(array)

    def get_n_with_energy(self, energy):
        return eigen_get_n_with_energy(self.cdata, energy)

    def save(self, file):
        data = np.ndarray((self.cdata.grid.n[1], self.cdata.grid.n[0]+1, self.cdata.grid.n[0]))
        data[:,:-1,:] = self.get_evec()
        data[:,-1,:] = self.get_eval()
        np.save(file, data)

    def load(self, file):
        data = np.load(file)
        self.get_evec()[:] = data[:,:-1,:]
        self.get_eval()[:] = data[:,-1,:]

cdef class GPSWorkspace:
    def __cinit__(self, ShGrid grid, Atom atom, double dt, double Emax):
        self.cdata = gps_ws_alloc(grid.data, atom.cdata, dt, Emax)

    def __dealloc__(self):
        gps_ws_free(self.cdata)

    def calc_s(self, Eigen eigen):
        gps_ws_calc_s(self.cdata, eigen.cdata)

    def prop(self, SWavefunc wf):
        gps_ws_prop(self.cdata, wf.cdata)

    def prop_comm(self, SWavefunc wf, Uabs uabs, Field field, double t):
        gps_ws_prop_common(self.cdata, wf.cdata, uabs.cdata, field.cdata, t)


cdef class SKnWorkspace:
    def __cinit__(self, ShGrid grid, Uabs uabs, int num_threads = -1):
        self.cdata = sh_workspace_alloc(
            grid.data,
            uabs.cdata,
            num_threads
        )
        self.uabs = uabs

    def __dealloc__(self):
        if self.cdata != NULL:
            sh_workspace_free(self.cdata)

    def prop(self, SWavefunc wf, Atom atom, Field field, double t, double dt):
        sh_workspace_prop(self.cdata, wf.cdata, atom.cdata, field.cdata, t, dt)

    def prop_img(self, SWavefunc wf, Atom atom, double dt):
        sh_workspace_prop_img(self.cdata, wf.cdata, atom.cdata, dt)


cdef class SOrbsWorkspace:
    def __cinit__(self, ShGrid sh_grid, SpGrid sp_grid, Uabs uabs, YlmCache ylm_cache, int Uxc_lmax = 3, int Uh_lmax = 3, Uxc uxc = UXC_LB, int num_threads=-1):
        assert(Uxc_lmax >= 0 and Uxc_lmax <= 3)
        assert(Uh_lmax >= 0 and Uh_lmax <= 3)

        self.cdata = sh_orbs_workspace_alloc(sh_grid.data, sp_grid.data, uabs.cdata, ylm_cache.cdata, Uh_lmax, Uxc_lmax, uxc.cdata, num_threads)
        self.uabs = uabs

    def __dealloc__(self):
        if self.cdata != NULL:
            sh_orbs_workspace_free(self.cdata)

    def prop_img(self, SOrbitals orbs, Atom atom, double dt):
        sh_orbs_workspace_prop_img(self.cdata, orbs.cdata, atom.cdata, dt)

    def prop(self, SOrbitals orbs, Atom atom, Field field, double t, double dt, bint calc_uee=True):
        sh_orbs_workspace_prop(self.cdata, orbs.cdata, atom.cdata, field.cdata, t, dt, calc_uee)

    def calc_uee(self, SOrbitals orbs, int Uxc_lmax=None, int Uh_lmax=None):
        if Uxc_lmax is None:
            Uxc_lmax = self.cdata.Uxc_lmax

        if Uh_lmax is None:
            Uh_lmax = self.cdata.Uh_lmax

        sh_orbs_workspace_calc_Uee(self.cdata, orbs.cdata, Uxc_lmax, Uh_lmax)
