import numpy as np
import h5py
import os

import tdse
from .task import CalcData, TaskAtom, CalcDataWithMask, TimeShapeMixin


class OrbShapeMixin(object):
    def get_shape(self, task):
        return (task.t.size, task.atom.countOrbs)

class UeeOrbData(CalcData):
    NAME = "uee"

    def __init__(self, dT, r_max = None, **kwargs):
        super().__init__(**kwargs)

        self.dT = dT
        self.r_max = r_max

    def get_shape(self, task: TaskAtom):
        if self.r_max == None or self.r_max >= task.r_max:
            self.Nr = task.Nr
        else:
            self.Nr = int(self.r_max/task.dr)

        self.dNt = int(self.dT/task.dt)
        self.Nt = (task.t.size // self.dNt) + 1

        return (self.Nt, 3, self.Nr)

    def calc(self, task, i, t):
        if task.rank == 0 and i % self.dNt == 0:
            self.dset[i // self.dNt] = task.ws.uee[:,:self.Nr]

class UpolOrbData(CalcData):
    NAME = "Upol"

    def __init__(self, is_average=True, dN=100, l=1, **kwargs):
        super().__init__(**kwargs)

        self.is_average = is_average
        self.dN = dN
        self.l = l

    def get_shape(self, task: TaskAtom):
        return (task.Nr,)

    def calc(self, task, i, t):
        if task.rank == 0:
            if self.is_average and (i+1) % self.dN == 0:
                count = task.t.size // self.dN
                E = task.field.E(t)
                self.dset[:] += task.ws.uee[self.l,:] / E**self.l / count
            elif i == task.t.size-1:
                E = task.field.E(t)
                self.dset[:] = task.ws.uee[self.l,:] / E**self.l

class AzOrbData(OrbShapeMixin, CalcData):
    NAME = "az"

    def calc_init(self, task, file):
        super().calc_init(task, file)
        if task.rank == 0:
            self.az_ne = np.zeros(self.dset.shape[1])
        else:
            self.az_ne = None

    def calc(self, task, i, t):
        tdse.calc.az_ne(task.orbs, task.atom_cache, task.field, t, az = self.az_ne)

        if task.rank == 0:
            self.dset[i] = self.az_ne

class CalcDataWithMask(CalcData):
    def __init__(self, mask=None, **kwargs):
        super().__init__(**kwargs)

        self.mask = mask

    def calc_init(self, task, file):
        super().calc_init(task, file)

        if self.mask is not None and task.rank == 0:
            self.mask.write_params(self.dset)

class NormOrbData(OrbShapeMixin, CalcDataWithMask):
    NAME = "n"

    def calc_init(self, task, file):
        super().calc_init(task, file)

        if task.rank == 0:
            self.n_ne = np.zeros(self.dset.shape[1])
        else:
            self.n_ne = None

    def calc(self, task, i, t):
        task.orbs.norm_ne(norm=self.n_ne, mask=self.mask)

        if task.rank == 0:
            self.dset[i] = self.n_ne


class ZOrbData(OrbShapeMixin, CalcDataWithMask):
    NAME = "z"

    def calc_init(self, task, file):
        super().calc_init(task, file)

        if task.rank == 0:
            self.z_ne = np.zeros(self.dset.shape[1])
        else:
            self.z_ne = None

    def calc(self, task, i, t):
        task.orbs.z_ne(mask=self.mask, z = self.z_ne)

        if task.rank == 0:
            self.dset[i] = self.z_ne


class OrbitalsTask(TaskAtom):
    atom = tdse.atom.Ar

    dt = 0.008
    dr = 0.025

    r_max = 100
    Nl = 2
    Nc = 33

    uxc = tdse.hartree_potential.UXC_LB
    Uxc_lmax = 1
    Uh_lmax = 3

    ground_state = None
    ground_state_task = None

    def __init__(self, path_res='res', mode=None, is_mpi=True, **kwargs):
        if self.ground_state_task is not None:
            self.dt    = self.ground_state_task.dt
            self.dr    = self.ground_state_task.dr
            self.r_max = self.ground_state_task.r_max
            self.uxc   = self.ground_state_task.uxc
            self.Nc    = self.ground_state_task.Nc
            self.atom  = self.ground_state_task.atom

        super().__init__(path_res, mode, is_mpi=is_mpi, **kwargs)

        self.sp_grid = tdse.grid.SpGrid(Nr=self.r_max/self.dr, Nc=self.Nc, Np=1, r_max=self.r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(self.Nl, self.sp_grid)

    def _get_state_filename(self, i):
        return os.path.join(self.save_path, 'orbs_{}.npy'.format(i))

    def save_state(self, i):
        self.orbs.save(self._get_state_filename(i))

    def load_state(self, i):
        self.orbs.load(self._get_state_filename(i))

    def calc_init(self):
        super().calc_init()

        self.orbs = tdse.orbitals.Orbitals(self.atom, self.sh_grid, self.comm)
        self.orbs.load(self.ground_state)

        self.ws = tdse.workspace.SOrbsWorkspace(self.atom_cache, self.sh_grid, self.sp_grid, self.uabs_cache, self.ylm_cache, Uxc_lmax=self.Uxc_lmax, Uh_lmax = self.Uh_lmax, uxc=self.uxc)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_prop(self, i, t):
        self.ws.prop(self.orbs, self.field, t, self.dt)

    def write_calc_params(self, params_grp: h5py.Group):
        super().write_calc_params(params_grp)

        params_grp.attrs['dt'] = self.dt
        params_grp.attrs['Nc'] = self.Nc
        params_grp.attrs['Uxc_Lmax'] = self.Uxc_lmax
        params_grp.attrs['Uh_Lmax'] = self.Uh_lmax

        self.uxc.write_params(params_grp)


class OrbitalsPolarizationTask(OrbitalsTask):
    Imax = 1e14

    Nl = 512
    Uh_lmax = 3

    freq = tdse.utils.length_to_freq(800, 'nm')
    dT = 0

    CALC_DATA = [
        UpolOrbData(name="upol_1", l=1),
        UpolOrbData(name="upol_2", l=2)
    ]

    class Field(tdse.field.FieldBase):
        def __init__(self, Imax, freq):
            self.E0 = tdse.utils.I_to_E(Imax)
            self.tp = np.pi/freq

        def _func(self, t):
            return self.E0*np.sin(0.5*np.pi*t/self.tp)**2

        @property
        def T(self):
            return self.tp

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.field = OrbitalsPolarizationTask.Field(self.Imax, self.freq)
        self.CALC_DATA.append(UeeOrbData(dT = self.field.T/10))


class OrbitalsGroundStateTask(TaskAtom):
    atom = tdse.atom.Ar

    dt = 0.008
    dr = 0.025

    T = 100
    r_max = 100
    Nc = 33

    uxc = tdse.hartree_potential.UXC_LB
    uabs = tdse.abs_pot.UabsZero()

    Uxc_lmax = 1
    Uh_lmax = 1

    CALC_DATA = ['orbs_gs', 'E', 'uee']

    def __init__(self, path_res='res', mode=None, is_mpi=False, **kwargs):
        self.Nl = self.atom.l_max + 1

        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.Nt = int(self.T / self.dt)

        self.sp_grid = tdse.grid.SpGrid(Nr=self.Nr, Nc=self.Nc, Np=1, r_max=self.r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(self.Nl, self.sp_grid)

    def calc_init(self):
        super().calc_init()

        self.ws = tdse.workspace.SOrbsWorkspace(self.atom_cache, self.sh_grid, self.sp_grid, self.uabs_cache, self.ylm_cache, Uxc_lmax=self.Uxc_lmax, Uh_lmax = self.Uh_lmax, uxc=self.uxc)

    def calc(self):
        self.calc_init()

        self.orbs, self.E = tdse.ground_state.orbs(self.atom, self.sh_grid, self.ws, self.dt, self.Nt, print_calc_info=True)
        self.orbs_gs = self.orbs.asarray()

        self.ws.calc_uee(self.orbs)
        self.uee = self.ws.uee[0]

        self.save()


class AzWfData(TimeShapeMixin, CalcData):
    NAME = "az"

    def calc(self, task, i, t):
        self.dset[i] = tdse.calc.az(task.wf, task.atom_cache, task.field, t)

class AzPolarizationWfData(TimeShapeMixin, CalcData):
    NAME = "az_pol"

    def calc_init(self, task, file):
        super().calc_init(task, file)

        self.u = task.orb_polarization_task.upol_1
        self.dudr = np.zeros(self.u.size)

        dr = task.sh_grid.dr
        self.dudr[0] = (self.u[1] - self.u[0])/dr
        self.dudr[-1] = (self.u[-1] - self.u[-2])/dr
        for i in range(1, self.dudr.size-1):
            self.dudr[i] = (self.u[i+1] - self.u[i-1])/(2*dr)

    def calc(self, task, i, t):
        self.dset[i] = tdse.calc.az_with_polarization(task.wf, task.atom_cache, self.u, self.dudr, task.field, t)

class NormWfData(TimeShapeMixin, CalcDataWithMask):
    NAME = "n"

    def calc(self, task, i, t):
        self.dset[i] = task.wf.norm(self.mask)

class ZWfData(TimeShapeMixin, CalcDataWithMask):
    NAME = "z"

    def calc(self, task, i, t):
        self.dset[i] = task.wf.z(self.mask)

class WfGroundStateTask(TaskAtom):
    atom = tdse.atom.H

    dt = 0.025
    dr = 0.125

    T = 100
    r_max = 100

    CALC_DATA = ['wf_gs', 'E']

    def __init__(self, path_res='res', mode=None, is_mpi=False, **kwargs):
        self.Nl = self.atom.l_max + 1
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.Nt = int(self.T / self.dt)

    def calc_init(self):
        super().calc_init()

        self.ws = tdse.workspace.SKnWorkspace(self.atom_cache, self.sh_grid, tdse.abs_pot.UabsZero())

    def calc(self):
        self.calc_init()

        self.wf, self.E = tdse.ground_state.wf(self.atom, self.sh_grid, self.ws, self.dt, self.Nt, **self.state)

        self.wf_gs = self.wf.asarray()
        self.save()


class WavefuncWithSourceTask(TaskAtom):
    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.grid_source = tdse.grid.ShGrid(Nr=self.sh_grid.Nr, Nl=self.atom.l_max+1, r_max=self.r_max)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def calc_init(self):
        super().calc_init()

        self.wf_source, self.Ip = self.calc_ground_state()
        self.wf = tdse.wavefunc.ShWavefunc(self.sh_grid)
        self.wf.asarray()[:] = 0.0

        self.ws = tdse.workspace.SKnWithSourceWorkspace(tdse.atom.AtomCache(tdse.atom.NONE, self.sh_grid), self.sh_grid, self.uabs_cache, self.wf_source, self.Ip)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = tdse.workspace.SKnWorkspace(self.atom_cache, self.grid_source, tdse.abs_pot.UabsCache(tdse.abs_pot.UabsZero(), self.grid_source))

        return tdse.ground_state.wf(self.atom, self.grid_source, ws, self.dt, 10000)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class WavefuncTask(TaskAtom):
    """
    """
    is_calc_ground_state = True

    Workspace = tdse.workspace.SKnWorkspace

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def create_workspace(self, uabs_cache = None):
        if uabs_cache is None:
            uabs_cache = self.uabs_cache

        return self.Workspace(self.atom_cache, self.sh_grid, uabs_cache)

    def calc_init(self):
        super().calc_init()

        self.ws = self.create_workspace()

        if self.is_calc_ground_state:
            print("Start calc ground state")
            self.wf = self.calc_ground_state(self.ws)
        else:
            self.wf = tdse.wavefunc.ShWavefunc(self.sh_grid)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = self.create_workspace(tdse.abs_pot.UabsZero())

        return tdse.ground_state.wf(self.atom, self.sh_grid, ws, self.dt, 10000)[0]

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class WavefuncNeTask(TaskAtom):
    Workspace = tdse.workspace.SKnNeWorkspace

    Rmin = 1e-3
    Ra   = 1.0

    AtomCacheClass = tdse.atom.AtomNeCache
    UabsCacheClass = tdse.abs_pot.UabsNeCache

    def create_grid(self):
        return tdse.grid.ShNeGrid(self.Rmin, self.r_max, self.Ra, self.dr, self.Nl)

class WfGpuTask(WavefuncTask):
    gpuGridNl = 1024
    threadsPerBlock = 8

    def __init__(self, **kwargs):
        tdse.calc.setGpuDevice(int(os.environ.get("CUDA_DEVICE", "0")))

        super().__init__(**kwargs)

    def create_workspace(self):
        return tdse.workspace_gpu.WfGPUWorkspace(self.atom_cache, self.sh_grid, self.uabs_cache,
                                                 self.gpuGridNl, self.threadsPerBlock)

    def calc_ground_state(self, ws=None):
        ws = super().create_workspace()
        self.wf_gs = tdse.ground_state.wf(self.atom, self.sh_grid, ws, self.dt, 10000)[0]

        return tdse.wavefunc_gpu.ShWavefuncGPU(self.wf_gs)

class WavefuncWithPolarization(WavefuncTask):
    orb_polarization_task = None

    Workspace = tdse.workspace.WfWithPolarizationWorkspace

    l_pol = 1

    def __init__(self, path_res='res', mode=None, **kwargs):
        self.orb_polarization_task.load()

        self.dt = self.orb_polarization_task.dt
        self.dr = self.orb_polarization_task.dr
        self.r_max = self.orb_polarization_task.r_max
        self.atom = self.orb_polarization_task.atom

        super().__init__(path_res, mode, **kwargs)

    def create_workspace(self, uabs_cache = None):
        if uabs_cache is None:
            uabs_cache = self.uabs_cache


        if self.l_pol == 1:
            return self.Workspace(self.atom_cache, self.sh_grid, uabs_cache, self.orb_polarization_task.upol_1[:])
        else:
            return self.Workspace(self.atom_cache, self.sh_grid, uabs_cache, self.orb_polarization_task.upol_1[:], self.orb_polarization_task.upol_2[:])
