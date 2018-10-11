import tdse

from .task import CalcData, TaskAtom, CalcDataWithMask, TimeShapeMixin


class AzWfData(TimeShapeMixin, CalcData):
    NAME = "az"

    def calc(self, task, i, t):
        self.dset[i] = tdse.calc.az(task.wf, task.atom_cache, task.field, t)

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

class WavefuncWithPolarization(WavefuncTask):
    orb_polarization_task = None

    Workspace = tdse.workspace.WfWithPolarizationWorkspace

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

        return self.Workspace(self.atom_cache, self.sh_grid, uabs_cache, self.orb_polarization_task.upol_1[:])
