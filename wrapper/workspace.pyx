import numpy as np
cimport numpy as np

from atom cimport Atom
from grid cimport ShGrid, SpGrid
from wavefunc cimport SWavefunc
from field cimport Field
from orbitals cimport SOrbitals
from sphere_harmonics cimport YlmCache


cdef class Eigen:
    def __cinit__(self, ShGrid grid):
        self.cdata = eigen_ws_alloc(grid.data)

    def __dealloc__(self):
        if self.cdata != NULL:
            eigen_ws_free(self.cdata)

    def calc(self, Atom atom):
        eigen_calc_for_atom(self.cdata, atom._data)

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
        self.cdata = gps_ws_alloc(grid.data, atom._data, dt, Emax)

    def __dealloc__(self):
        gps_ws_free(self.cdata)

    def calc_s(self, Eigen eigen):
        gps_ws_calc_s(self.cdata, eigen.cdata)

    def prop(self, SWavefunc wf):
        gps_ws_prop(self.cdata, wf.data)

    def prop_comm(self, SWavefunc wf, Uabs uabs, Field field, double t):
        gps_ws_prop_common(self.cdata, wf.data, uabs.cdata, field.data, t)


cdef class SKnWorkspace:
    def __cinit__(self, ShGrid grid, Uabs uabs, int num_threads = -1):
        self.data = sh_workspace_alloc(
            grid.data,
            uabs.cdata,
            num_threads
        )
        self.uabs = uabs

    def __dealloc__(self):
        if self.data != NULL:
            sh_workspace_free(self.data)

    def prop(self, SWavefunc wf, Atom atom, Field E, double t, double dt):
        sh_workspace_prop(self.data, wf.data, atom._data, E.data, t, dt)

    def prop_img(self, SWavefunc wf, Atom atom, double dt):
        sh_workspace_prop_img(self.data, wf.data, atom._data, dt)


cdef class SOrbsWorkspace:
    def __cinit__(self, ShGrid sh_grid, SpGrid sp_grid, Uabs uabs, YlmCache ylm_cache, int num_threads=-1):
        self._data = sh_orbs_workspace_alloc(sh_grid.data, sp_grid.data, uabs.cdata, ylm_cache._data, num_threads)
        self.uabs = uabs

    def __dealloc__(self):
        if self._data != NULL:
            sh_orbs_workspace_free(self._data)

    def prop_img(self, SOrbitals orbs, Atom atom, double dt):
        sh_orbs_workspace_prop_img(self._data, orbs._data, atom._data, dt)

    def prop(self, SOrbitals orbs, Atom atom, Field f, double t, double dt):
        sh_orbs_workspace_prop(self._data, orbs._data, atom._data, f.data, t, dt)
