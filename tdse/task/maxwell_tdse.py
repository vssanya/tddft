import tdse
import tdse.maxwell

import numpy as np

from .wf_array_task import CalcData, WavefuncArrayGPUTask


class EData(CalcData):
    NAME = "E"

    def __init__(self, x, **kwargs):
        super().__init__(**kwargs)
        
        self.x = x

    def get_shape(self, task):
        return (task.t.size//self.m_dt)

    def calc_init(self, task, file):
        super().calc_init(task, file)

        self.x_index = int(self.x / self.m_grid.d)

    def calc(self, task, i, t):
        if i % self.m_dt:
            self.dset[i // self.m_dt] = task.maxwell_ws.E[self.x_index]

class MaxwellTDSETask(WavefuncArrayGPUTask):
    dx = tdse.utils.unit_to(20, "nm") # step in Maxwell equation
    L = tdse.utils.unit_to(200e3, "nm") # length of media
    ksi = 0.9 # maxwell propogation parameter

    x0 = tdse.utils.unit_to(50, "nm") # init location of center laser pulse

    n = None # Gas concentration

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, **kwargs)

        self.m_grid = tdse.grid.Grid1d(int(self.L/self.dx), self.dx)
        self.n = np.zeros(self.m_grid.N)
        self.x = np.linspace(0, self.L, self.n.size)

    def calc_init(self):
        self.calc_n(self.x)

        self.N_index = np.argwhere(self.n != 0.0).reshape(-1)
        self.N = self.N_index.size

        self.az = np.zeros((2,self.N))
        self.vz = np.zeros((2,self.N))
        self.z  = np.zeros((2,self.N))

        self.maxwell_ws = tdse.maxwell.MaxwellWorkspace1D(self.m_grid)
        self.m_dt = int(self.dt / self.maxwell_ws.get_dt(self.ksi))

        E = self.field.E(self.field.T/2 - (self.x - self.x0)/tdse.const.C)
        self.maxwell_ws.E[:] = E
        self.maxwell_ws.D[:] = E

        H = self.field.E(self.field.T/2 - self.maxwell_ws.get_dt(self.ksi) - (self.x - self.x0 + self.m_grid.d/2)/tdse.const.C)
        self.maxwell_ws.H[:] = H

        self.P = np.zeros(self.m_grid.N)

        super().calc_init()

    def calc_prop(self, i, t):
        super().calc_prop(i, t)

        self.az[0] = self.az[1]
        self.vz[0] = self.vz[1]
        self.z[0] = self.z[1]

        tdse.calc_gpu.az(self.wf_array, self.atom_cache, self.E, self.az[1])

        self.vz[1] = self.vz[0] + (self.az[0] + self.az[1])*self.dt/2
        self.z[1]  = self.z[0]  + (self.vz[0] + self.vz[1])*self.dt/2

        if i % self.m_dt == 0:
            self.P[self.N_index] = - self.z[1]*self.n[self.N_index]
            self.maxwell_ws.prop(self.dt*self.m_dt, pol=self.P)

    def calc_field(self, t):
        self.E[:] = self.maxwell_ws.E[self.N_index]

    def calc_n(self, x):
        pass
