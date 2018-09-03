import tdse

from .task import Task


class TdsfmInnerTask(Task):
    CALC_DATA = ['p',]

    sp_grid = None
    l_min = 0
    l_max = -1

    without_ground_state = False

    def __init__(self, tdsfm_task, **kwargs):
        super().__init__(is_mpi=False, **kwargs)

        self.task = tdsfm_task

    def calc_init(self):
        self.task.calc_init()
        self.task.load()
        self.task.wf.asarray()[:] = self.task.psi[:]

        if self.sp_grid is None:
            self.tdsfm = self.task.tdsfm
        else:
            self.tdsfm = tdse.tdsfm.TDSFM_VG(self.sp_grid, self.task.sh_grid, 0)

        self.p = self.tdsfm.asarray()

    def calc(self):
        self.calc_init()

        if (self.without_ground_state):
            self.task.wf.asarray()[:] = self.task.wf.asarray()[:] - (self.task.wf*self.task.wf_gs)*self.task.wf_gs.asarray()[:]

        self.tdsfm.calc_inner(self.task.field, self.task.wf, 0, 0, self.task.sh_grid.Nr, self.l_min, self.l_max)

        self.save()
