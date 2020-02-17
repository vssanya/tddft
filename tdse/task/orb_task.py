import numpy as np
import h5py

import tdse

from .task import CalcData, CalcDataWithMask, TaskAtom


class OrbShapeMixin(object):
    def get_shape(self, task):
        return (task.t.size, task.atom.countOrbs)

class UeeOrbData(CalcData):
    NAME = "uee"

    def __init__(self, dT, r_max = None, r_core = None, **kwargs):
        super().__init__(**kwargs)

        self.dT = dT
        self.r_max = r_max
        self.r_core = r_core
        self.range = None

    def calc_init(self, task, file):
        if self.r_max == None or self.r_max >= task.r_max:
            self.Nr = task.sh_grid.Nr
        else:
            self.Nr = int(self.r_max/task.dr)

        self.dNt = int(self.dT/task.dt)
        self.Nt = (task.t.size // self.dNt) + 1

        if self.r_core is not None:
            self.range = task.sh_grid.getRange(self.r_core)

        super().calc_init(task, file)

    def get_shape(self, task: TaskAtom):
        return (self.Nt, 3, self.Nr)

    def calc(self, task, i, t):
        if i % self.dNt == 0:
            if self.range is None:
                task.ws.calc_uee(task.orbs, rRange = self.range)

            if task.rank == 0:
                self.dset[i // self.dNt] = task.ws.uee[:,:self.Nr]

class UpolOrbData(CalcData):
    NAME = "Upol"

    def __init__(self, is_average=True, dN=100, l=1, **kwargs):
        super().__init__(**kwargs)

        self.is_average = is_average
        self.dN = dN
        self.l = l

    def get_shape(self, task: TaskAtom):
        return (task.sh_grid.Nr,)

    def calc(self, task, i, t):
        if task.rank == 0:
            if self.is_average and (i+1) % self.dN == 0:
                count = task.t.size // self.dN
                E = task.field.E(t)
                self.dset[:] += task.ws.uee[self.l,:] / E**self.l / count
            elif i == task.t.size-1:
                E = task.field.E(t)
                self.dset[:] = task.ws.uee[self.l,:] / E**self.l

class NspOrbData(CalcData):
    NAME = "Nsp"

    def __init__(self, dT, **kwargs):
        super().__init__(**kwargs)

        self.dT = dT

    def calc_init(self, task, file: h5py.File):
        self.sp_grid = task.sp_grid

        self.dNt = int(self.dT/task.dt)
        self.Nt = (task.t.size // self.dNt) + 1

        if task.rank == 0:
            self.n_tmp = np.zeros(self.sp_grid.shape)
        else:
            self.n_tmp = None

        super().calc_init(task, file)


    def get_shape(self, task):
        return (self.Nt, self.sp_grid.Nc, self.sp_grid.Nr)

    def calc(self, task, i, t):
        if i % self.dNt == 0:
            task.orbs.n_sp(task.sp_grid, task.ylm_cache, self.n_tmp)

            if task.rank == 0:
                self.dset[i // self.dNt] = self.n_tmp


class PsiOrbData(CalcData):
    NAME = "orbs"
    DTYPE = np.cdouble

    def __init__(self, dT, Nl=-1, **kwargs):
        super().__init__(**kwargs)

        self.dT = dT
        self.Nl = Nl

    def calc_init(self, task, file: h5py.File):
        self.sh_grid = task.sh_grid
        self.countOrbs = task.atom.countOrbs

        self.dNt = int(self.dT/task.dt)
        self.Nt = (task.t.size // self.dNt) + 1

        if self.Nl == -1:
            self.Nl = self.sh_grid.Nl

        if task.rank == 0:
            self.orbs_tmp = np.zeros((self.countOrbs, self.Nl, self.sh_grid.Nr), dtype=np.complex)
        else:
            self.orbs_tmp = None

        super().calc_init(task, file)


    def get_shape(self, task):
        return (self.Nt, self.countOrbs, self.Nl, self.sh_grid.Nr)

    def calc(self, task, i, t):
        if i % self.dNt == 0:
            task.orbs.collect(self.orbs_tmp, self.Nl)

            if task.rank == 0:
                self.dset[i // self.dNt] = self.orbs_tmp


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

class AzVeeOrbData(OrbShapeMixin, CalcData):
    NAME = "az_vee"

    def __init__(self, l=0, **kwargs):
        super().__init__(**kwargs)

        self.l = l

    def calc_init(self, task, file):
        super().calc_init(task, file)
        if task.rank == 0:
            self.az_ne = np.zeros(self.dset.shape[1])
        else:
            self.az_ne = None

        self.Uee = task.ws.uee
        self.dUee_dr = np.zeros_like(self.Uee)

    def calc(self, task, i, t):
        tdse.calc.az_ne_Vee(task.orbs, task.atom_cache, task.field, t, self.Uee, self.dUee_dr, az = self.az_ne, l = self.l)

        if task.rank == 0:
            self.dset[i] = self.az_ne

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
    dt_count = None

    dr = 0.025

    r_max = 100
    Nl = 2
    Nc = 33

    uxc = tdse.hartree_potential.UXC_LB
    Uxc_lmax = 1
    Uh_lmax = 3

    useTwoPointUeeCalcScheme = False

    ground_state = None
    ground_state_task = None

    Workspace = tdse.workspace.ShOrbitalsWS
    Orbitals = tdse.orbitals.ShOrbitals

    active_orbs = None
    rank_orbs = None

    def init_from_ground_state_task(self, gs_task):
        super().init_from_ground_state_task(gs_task)

        self.uxc   = gs_task.uxc
        self.Nc    = gs_task.Nc

    def __init__(self, path_res='res', mode=None, is_mpi=True, **kwargs):
        super().__init__(path_res, mode, is_mpi=is_mpi, **kwargs)

        self.sp_grid = tdse.grid.SpGrid(Nr=self.sh_grid.Nr, Nc=self.Nc, Np=1, r_max=self.r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(self.Nl, self.sp_grid)

    def _get_state_filename(self, i):
        return os.path.join(self.save_path, 'orbs_{}.npy'.format(i))

    def save_state(self, i):
        self.orbs.save(self._get_state_filename(i))

    def load_state(self, i):
        self.orbs.load(self._get_state_filename(i))

    def calc_init(self):
        super().calc_init()

        self.orbs = self.Orbitals(self.atom, self.sh_grid, self.comm, self.rank_orbs)
        self.orbs.load(self.ground_state)

        self.ws = self.Workspace(self.atom_cache, self.sh_grid, self.sp_grid, self.uabs_cache, self.ylm_cache, Uxc_lmax=self.Uxc_lmax, Uh_lmax = self.Uh_lmax, uxc=self.uxc)
        if self.useTwoPointUeeCalcScheme:
            self.ws.set_time_approx_uee_two_point(self.orbs)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_prop(self, i, t):
        self.ws.prop(self.orbs, self.field, t, self.dt, active_orbs = self.active_orbs, dt_count = self.dt_count)

    def write_calc_params(self, params_grp: h5py.Group):
        super().write_calc_params(params_grp)

        params_grp.attrs['dt'] = self.dt
        params_grp.attrs['Nc'] = self.Nc
        params_grp.attrs['Uxc_Lmax'] = self.Uxc_lmax
        params_grp.attrs['Uh_Lmax'] = self.Uh_lmax

        self.uxc.write_params(params_grp)

class OrbitalsNeTask(OrbitalsTask):
    Workspace = tdse.workspace.ShNeOrbitalsWS
    Orbitals = tdse.orbitals.ShNeOrbitals

    Rmin = 1e-3
    Ra   = 1.0

    AtomCacheClass = tdse.atom.ShNeAtomCache
    UabsCacheClass = tdse.abs_pot.UabsNeCache

    def __init__(self, path_res='res', mode=None, is_mpi=True, **kwargs):
        if self.ground_state_task is not None:
            self.Rmin  = self.ground_state_task.Rmin
            self.Ra    = self.ground_state_task.Ra

        super().__init__(path_res, mode, is_mpi=is_mpi, **kwargs)

    def create_grid(self):
        return tdse.grid.ShNeGrid(self.Rmin, self.r_max, self.Ra, self.dr, self.Nl)

class UeeCalcOnceMixin():
    def calc_prop(self, i, t):
        if i == 0:
            calc_uee = True
        else:
            calc_uee = False

        self.ws.prop(self.orbs, self.field, t, self.dt, active_orbs = self.active_orbs, calc_uee=calc_uee, dt_count = self.dt_count)

class OrbitalsNeWithoutFieldTask(OrbitalsNeTask):
    def calc_prop(self, i, t):
        self.ws.prop_ha(self.orbs, self.dt);

class OrbitalsPolarizationTask(OrbitalsTask):

    class Field(tdse.field.FieldBase):
        def __init__(self, Imax, freq):
            self.E0 = tdse.utils.I_to_E(Imax)
            self.tp = np.pi/freq

        def _func(self, t):
            if t < self.tp:
                return self.E0*np.sin(0.5*np.pi*t/self.tp)**2
            else:
                return 0.0

        @property
        def T(self):
            return self.tp

    Imax = 1e14

    Nl = 512
    Uh_lmax = 3

    freq = tdse.utils.length_to_freq(800, 'nm')
    dT = 0

    CALC_DATA = [
        UpolOrbData(name="upol_1", l=1),
        UpolOrbData(name="upol_2", l=2),
        ZOrbData(r_core=4, dr=2)
    ]

    class Field(tdse.field.FieldBase):
        def __init__(self, Imax, freq):
            self.E0 = tdse.utils.I_to_E(Imax)
            self.tp = np.pi/freq

        def _func(self, t):
            if t < self.tp:
                return self.E0*np.sin(0.5*np.pi*t/self.tp)**2
            else:
                return 0.0

        @property
        def T(self):
            return self.tp

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.field = OrbitalsPolarizationTask.Field(self.Imax, self.freq)
        self.CALC_DATA.append(UeeOrbData(dT = self.field.T/10))


class OrbitalsPolarizationNeTask(OrbitalsNeTask):
    Imax = 1e14
    N = 10

    Nl = 512
    Uh_lmax = 3

    freq = tdse.utils.length_to_freq(800, 'nm')
    dT = 0

    r_core = 4

    CALC_DATA = [
        UpolOrbData(name="upol_1", l=1),
        UpolOrbData(name="upol_2", l=2)
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.field = OrbitalsPolarizationTask.Field(self.Imax, self.freq)

        self.CALC_DATA.append(UeeOrbData(dT = self.field.T/self.N))
        if self.r_core is not None:
            self.CALC_DATA.append(ZOrbData(r_core=self.r_core, dr=self.r_core/10))
            self.CALC_DATA.append(UeeOrbData(name="uee_rcore", dT = self.field.T/self.N, r_core=self.r_core))


class OrbitalsGroundStateTask(TaskAtom):
    atom = tdse.atom.Ar

    dt = 0.008
    dt_count = None

    dr = 0.025

    T = 100
    r_max = 100
    Nc = 33

    uxc = tdse.hartree_potential.UXC_LB
    uabs = tdse.abs_pot.UabsZero()

    Uxc_lmax = 1
    Uh_lmax = 1

    Workspace = tdse.workspace.ShOrbitalsWS
    Orbitals = tdse.orbitals.ShOrbitals

    FUNCS = {
        'GroundStateSearchFunc': tdse.ground_state.orbs
    }

    CALC_DATA = ['orbs_gs', 'E', 'uee']

    def __init__(self, path_res='res', mode=None, is_mpi=False, **kwargs):
        self.Nl = self.atom.l_max + 1

        super().__init__(path_res, mode, is_mpi=False, **kwargs)

        self.Nt = int(self.T / self.dt)

        self.sp_grid = tdse.grid.SpGrid(Nr=self.sh_grid.Nr, Nc=self.Nc, Np=1, r_max=self.r_max)
        self.ylm_cache = tdse.sphere_harmonics.YlmCache(self.Nl, self.sp_grid)

    def calc_init(self):
        super().calc_init()

        self.ws = self.Workspace(self.atom_cache, self.sh_grid, self.sp_grid, self.uabs_cache, self.ylm_cache, Uxc_lmax=self.Uxc_lmax, Uh_lmax = self.Uh_lmax, uxc=self.uxc)

    def calc(self):
        self.calc_init()

        self.orbs, self.E = self.FUNCS['GroundStateSearchFunc'](self.atom, self.sh_grid, self.ws, self.dt, self.Nt, self.Orbitals, self.AtomCacheClass, True, dt_count=self.dt_count)
        self.orbs_gs = self.orbs.asarray()

        self.ws.calc_uee(self.orbs)
        self.uee = self.ws.uee[0]

        self.save()


class OrbitalsGroundStateNeTask(OrbitalsGroundStateTask):
    Workspace = tdse.workspace.ShNeOrbitalsWS
    Orbitals = tdse.orbitals.ShNeOrbitals

    Rmin = 1e-3
    Ra   = 1.0

    AtomCacheClass = tdse.atom.ShNeAtomCache
    UabsCacheClass = tdse.abs_pot.UabsNeCache

    def create_grid(self):
        return tdse.grid.ShNeGrid(self.Rmin, self.r_max, self.Ra, self.dr, self.Nl)
