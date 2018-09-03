import tdse

from .task import Task


class SFATask(Task):
    Np = 100
    Ntheta = 50

    p_max = 4

    field = None

    dt = 0.025

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, is_mpi=False, **kwargs)
        self.grid = tdse.grid.SpGrid2d(self.Np, self.Ntheta, self.p_max)

    def calc_init(self):
        self.ws = tdse.workspace.SFAWorkspace()
        self.wf = tdse.wavefunc.CtWavefunc(self.grid)

        self.t = self.field.get_t(self.dt, dT=self.dT)

    def calc_prop(self, i, t):
        self.ws.prop(self.wf, self.field, t, self.dt)


