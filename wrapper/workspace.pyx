import numpy as np
cimport numpy as np

from atom cimport AtomCache
from grid cimport ShGrid, SpGrid
from wavefunc cimport ShWavefunc, CtWavefunc
from field cimport Field
from orbitals cimport Orbitals
from sphere_harmonics cimport YlmCache
from hartree_potential cimport Uxc
from hartree_potential import UXC_LB


cdef class Eigen:
    def __cinit__(self, ShGrid grid):
        self.cdata = eigen_ws_alloc(grid.data)

    def __dealloc__(self):
        if self.cdata != NULL:
            eigen_ws_free(self.cdata)

    def calc(self, AtomCache atom):
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
        self.cdata = ws_gps_alloc(grid.data, atom.cdata, dt, Emax)

    def __dealloc__(self):
        ws_gps_free(self.cdata)

    def calc_s(self, Eigen eigen):
        ws_gps_calc_s(self.cdata, eigen.cdata)

    def prop(self, ShWavefunc wf):
        ws_gps_prop(self.cdata, wf.cdata)

    def prop_comm(self, ShWavefunc wf, UabsCache uabs, Field field, double t):
        ws_gps_prop_common(self.cdata, wf.cdata, uabs.cdata, field.cdata, t)


cdef class SFAWorkspace:
    def __cinit__(self):
        self.cdata = new momentum_space()

    def __dealloc__(self):
        del self.cdata

    def prop(self, CtWavefunc wf, Field field, double t, double dt):
        self.cdata.propagate(wf.cdata[0], field.cdata, t, dt)

cdef class SKnWorkspace:
    def __cinit__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, int num_threads = -1, int propType=4):
        cdef PropAtType propAtType
        if propType == 4:
            propAtType = Odr4
        else:
            propAtType = Odr3

        self.cdata = new WfBase(grid.data[0], atom_cache.cdata[0], uabs.cdata[0], propAtType, num_threads)

        self.uabs = uabs
        self.atom_cache = atom_cache

    def __dealloc__(self):
        del self.cdata

    def __init__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, int num_threads = -1):
        pass

    def prop(self, ShWavefunc wf, Field field, double t, double dt):
        self.cdata.prop(wf.cdata[0], field.cdata, t, dt)

    def prop_abs(self, ShWavefunc wf, double dt):
        self.cdata.prop_abs(wf.cdata[0], dt)

    def prop_without_field(self, ShWavefunc wf, double dt):
        self.cdata.prop_without_field(wf.cdata[0], dt)

    def prop_img(self, ShWavefunc wf, double dt):
        self.cdata.prop_img(wf.cdata[0], dt)


cdef class SKnAWorkspace:
    def __cinit__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, int num_threads = -1, int propType=4):
        cdef PropAtType propAtType
        if propType == 4:
            propAtType = Odr4
        else:
            propAtType = Odr3

        self.cdata = new WfA(grid.data[0], atom_cache.cdata[0], uabs.cdata[0], propAtType, num_threads)
        self.uabs = uabs
        self.atom_cache = atom_cache

    def __dealloc__(self):
        del self.cdata

    def __init__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, int num_threads = -1):
        pass

    def prop(self, ShWavefunc wf, Field field, double t, double dt):
        self.cdata.prop(wf.cdata[0], field.cdata, t, dt)

    def prop_without_field(self, ShWavefunc wf, double dt):
        self.cdata.prop_without_field(wf.cdata[0], dt)

    def prop_img(self, ShWavefunc wf, double dt):
        self.cdata.prop_img(wf.cdata[0], dt)

cdef class WfWithPolarizationWorkspace:
    def __cinit__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, double[:] Upol_1, double[:] Upol_2 = None, int num_threads = -1):
        if Upol_2 is None:
            self.cdata = new WfWithPolarization(grid.data[0], atom_cache.cdata[0], uabs.cdata[0], &Upol_1[0], NULL, Odr4, num_threads)
        else:
            self.cdata = new WfWithPolarization(grid.data[0], atom_cache.cdata[0], uabs.cdata[0], &Upol_1[0], &Upol_2[0], Odr4, num_threads)

        self.uabs = uabs
        self.atom_cache = atom_cache

    def __dealloc__(self):
        del self.cdata

    def __init__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, double[:] Upol_1, double[:] Upol_2 = None, int num_threads = -1):
        self.Upol_1 = Upol_1
        self.Upol_2 = Upol_2

    def prop(self, ShWavefunc wf, Field field, double t, double dt):
        self.cdata.prop(wf.cdata[0], field.cdata, t, dt)

    def prop_without_field(self, ShWavefunc wf, double dt):
        self.cdata.prop_without_field(wf.cdata[0], dt)

    def prop_img(self, ShWavefunc wf, double dt):
        self.cdata.prop_img(wf.cdata[0], dt)

cdef class SKnWithSourceWorkspace:
    def __cinit__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, ShWavefunc source, double E, int num_threads = -1):
        self.cdata = new WfEWithSource(grid.data[0], atom_cache.cdata[0], uabs.cdata[0], source.cdata[0], E, Odr4, num_threads)
        self.uabs = uabs
        self.source = source
        self.atom_cache = atom_cache

    def __dealloc__(self):
        del self.cdata

    def __init__(self, AtomCache atom_cache, ShGrid grid, UabsCache uabs, ShWavefunc source, int num_threads = -1):
        pass

    def prop(self, ShWavefunc wf, Field field, double t, double dt):
        self.cdata.prop(wf.cdata[0], field.cdata, t, dt)

    @property
    def abs_norm(self):
        return self.cdata.abs_norm


cdef class SOrbsWorkspace:
    def __cinit__(self, AtomCache atom_cache, ShGrid sh_grid, SpGrid sp_grid, UabsCache uabs, YlmCache ylm_cache, int Uxc_lmax = 3, int Uh_lmax = 3, Uxc uxc = UXC_LB, int num_threads=-1):
        assert(Uxc_lmax >= 0 and Uxc_lmax <= 3)
        assert(Uh_lmax >= 0 and Uh_lmax <= 3)

        self.cdata = new orbs(sh_grid.data[0], sp_grid.data[0],
                              atom_cache.cdata[0], uabs.cdata[0],
                              ylm_cache.cdata[0], Uh_lmax, Uxc_lmax,
                              uxc.cdata, Odr4, num_threads)
        self.uabs = uabs
        self.atom_cache = atom_cache

    def __dealloc__(self):
        del self.cdata

    def __init__(self, AtomCache atom_cache, ShGrid sh_grid, SpGrid sp_grid, UabsCache uabs, YlmCache ylm_cache, int Uxc_lmax = 3, int Uh_lmax = 3, Uxc uxc = UXC_LB, int num_threads=-1):
        pass

    def prop_img(self, Orbitals orbs, double dt):
        self.cdata.prop_img(orbs.cdata[0], dt)

    def prop_ha(self, Orbitals orbs, double dt):
        self.cdata.prop_ha(orbs.cdata[0], dt)

    def prop(self, Orbitals orbs, Field field, double t, double dt, bint calc_uee=True):
        self.cdata.prop(orbs.cdata[0], field.cdata, t, dt, calc_uee)

    def calc_uee(self, Orbitals orbs, Uxc_lmax=None, Uh_lmax=None):
        if Uxc_lmax is None:
            Uxc_lmax = self.cdata.Uxc_lmax

        if Uh_lmax is None:
            Uh_lmax = self.cdata.Uh_lmax

        self.cdata.calc_Uee(orbs.cdata[0], Uxc_lmax, Uh_lmax)

    @property
    def uee(self):
        cdef double[:,::1] res = <double[:3,:self.cdata.sh_grid.n[0]]>self.cdata.Uee
        return np.asarray(res)

    @property
    def n_sp(self):
        cdef double[:,::1] res = <double[:self.cdata.sp_grid.n[1],:self.cdata.sp_grid.n[0]]>self.cdata.n_sp
        return np.asarray(res)
