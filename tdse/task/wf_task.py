import numpy as np
import h5py
import os

import tdse
from .task import CalcData, TaskAtom, CalcDataWithMask, TimeShapeMixin


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

class FinWfData(CalcData):
    NAME = "psi_final"
    DTYPE = np.complex

    def get_shape(self, task):
        return task.wf.asarray().shape

    def calc_finish(self, task):
        self.dset[:] = task.wf.asarray()[:]

class NormWfData(TimeShapeMixin, CalcDataWithMask):
    NAME = "n"

    def calc(self, task, i, t):
        self.dset[i] = task.wf.norm(self.mask)

class ZWfData(TimeShapeMixin, CalcDataWithMask):
    NAME = "z"

    def calc(self, task, i, t):
        self.dset[i] = task.wf.z(self.mask)

class Z2WfData(TimeShapeMixin, CalcDataWithMask):
    NAME = "z2"

    def calc(self, task, i, t):
        self.dset[i] = task.wf.z2(self.mask)

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

        self.ws = tdse.workspace.ShWavefuncWS(self.atom_cache, self.sh_grid, tdse.abs_pot.UabsZero())

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

        self.ws = tdse.workspace.SKnWithSourceWorkspace(tdse.atom.ShAtomCache(tdse.atom.NONE, self.sh_grid), self.sh_grid, self.uabs_cache, self.wf_source, self.Ip)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = tdse.workspace.ShWavefuncWS(self.atom_cache, self.grid_source, tdse.abs_pot.UabsZeroCache(self.grid_source))

        return tdse.ground_state.wf(self.atom, self.grid_source, ws, self.dt, 100)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class WavefuncTask(TaskAtom):
    """
    """
    is_calc_ground_state = True
    ground_state = None
    ground_state_dt = None
    ground_state_T = 100

    Workspace = tdse.workspace.ShWavefuncWS
    Wavefunc = tdse.wavefunc.ShWavefunc

    prop_type = 4

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def create_workspace(self, uabs_cache = None):
        if uabs_cache is None:
            uabs_cache = self.uabs_cache

        return self.Workspace(self.atom_cache, self.sh_grid, uabs_cache, propType = self.prop_type)

    def calc_init(self):
        super().calc_init()

        self.ws = self.create_workspace()

        if self.is_calc_ground_state and not (type(self.ground_state) is np.ndarray):
            print("Start calc ground state")
            self.wf, self.Ip = self.calc_ground_state(self.ws)
        else:
            self.wf = self.Wavefunc(self.sh_grid, self.atom.ground_state.m)
            if type(self.ground_state) is np.ndarray:
                psi = self.wf.asarray()
                psi[:] = 0.0
                psi[self.atom.ground_state.l, :self.ground_state.size] = self.ground_state

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_ground_state(self, ws=None):
        if ws is None:
            ws = self.create_workspace(tdse.abs_pot.UabsZeroCache(self.sh_grid))

        if self.ground_state_dt is None:
            self.ground_state_dt = self.dt

        return tdse.ground_state.wf(self.atom, self.sh_grid, ws, self.ground_state_dt,
                int(self.ground_state_T/self.ground_state_dt), self.Wavefunc, ground_state = self.ground_state)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)

class WavefuncNeTask(WavefuncTask):
    Workspace = tdse.workspace.ShNeWavefuncWS
    Wavefunc = tdse.wavefunc.ShNeWavefunc

    Rmin = 1e-3
    Ra   = 1.0

    AtomCacheClass = tdse.atom.ShNeAtomCache
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
        self.wf_gs, Ip = tdse.ground_state.wf(self.atom, self.sh_grid, ws, self.dt, 100)

        return tdse.wavefunc_gpu.ShWavefuncGPU(self.wf_gs), Ip

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
