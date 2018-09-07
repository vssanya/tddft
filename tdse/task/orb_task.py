import numpy as np
import h5py

import tdse

from .task import CalcData, CalcDataWithMask, TaskAtom


class OrbShapeMixin(object):
    def get_shape(self, task):
        return (task.t.size, task.atom.countOrbs)

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

