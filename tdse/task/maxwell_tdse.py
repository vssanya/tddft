import tdse
import tdse.maxwell

import numpy as np

from .wf_array_task import CalcData, WavefuncArrayGPUTask, Task


class EdepsTData(CalcData):
    NAME = "Et"

    def __init__(self, x, **kwargs):
        super().__init__(**kwargs)
        
        self.x = x

    def get_shape(self, task):
        return (task.t.size//task.m_dt,)

    def calc_init(self, task, file):
        super().calc_init(task, file)

        self.x_index = int(self.x / task.m_grid.d)

    def calc(self, task, i, t):
        if i % task.m_dt:
            self.dset[i // task.m_dt] = task.maxwell_ws.E[self.x_index]

class EdepsXData(CalcData):
    NAME = "Ex"

    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        
        self.t = t

    def get_shape(self, task):
        return (self.t.size, task.m_grid.N)

    def calc_init(self, task, file):
        super().calc_init(task, file)

        self.t_index = (self.t / task.dt).astype(int)

    def calc(self, task, i, t):
        if np.any(self.t_index == i):
            print("Data: ", np.argwhere(self.t_index == i))
            self.dset[np.argwhere(self.t_index == i),:] = task.maxwell_ws.E[:]

class MaxwellTDSETask(WavefuncArrayGPUTask):
    dx = tdse.utils.unit_to(20, "nm") # step in Maxwell equation
    L = tdse.utils.unit_to(200e3, "nm") # length of media
    ksi = 0.9 # maxwell propogation parameter

    x0 = tdse.utils.unit_to(60, "nm") # init location of center laser pulse

    Imin = 0.0 # Minimum intensity to start calculating the medium response

    n = None # Gas concentration

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, **kwargs)

        self.m_grid = tdse.grid.Grid1d(int(self.L/self.dx), self.dx)
        self.n = np.zeros(self.m_grid.N)
        self.x = np.linspace(0, self.L, self.n.size)

        self.wait_pulse = True

    def calc_init(self):
        self.calc_n(self.x)

        self.N_index = np.argwhere(self.n != 0.0).reshape(-1)
        self.N = self.N_index.size

        self.az = np.zeros((2,self.N))
        self.vz = np.zeros((2,self.N))
        self.z  = np.zeros((2,self.N))

        self.maxwell_ws = tdse.maxwell.MaxwellWorkspace1D(self.m_grid)
        self.m_dt = int(self.maxwell_ws.get_dt(self.ksi) / self.dt)
        print("m_dt = {}".format(self.m_dt))

        E = self.field.E(self.field.T/2 - (self.x - self.x0)/tdse.const.C)
        self.maxwell_ws.E[:] = E
        self.maxwell_ws.D[:] = E

        H = self.field.E(self.field.T/2 - self.maxwell_ws.get_dt(self.ksi)/2 - (self.x - self.x0 + self.m_grid.d/2)/tdse.const.C)
        self.maxwell_ws.H[:] = H

        self.P = np.zeros(self.m_grid.N)

        super().calc_init()

    def calc_prop(self, i, t):
        if not self.wait_pulse:
            super().calc_prop(i, t)

            self.az[0] = self.az[1]
            self.vz[0] = self.vz[1]
            self.z[0] = self.z[1]

            tdse.calc_gpu.az(self.wf_array, self.atom_cache, self.E, self.az[1])

            self.vz[1] = self.vz[0] + (self.az[0] + self.az[1])*self.dt/2
            self.z[1]  = self.z[0]  + (self.vz[0] + self.vz[1])*self.dt/2

        if i % self.m_dt == 0:
            self.P[self.N_index] = - self.z[1]*self.n[self.N_index]
            self.maxwell_ws.prop(dt=self.dt*self.m_dt, pol=self.P)

            if self.wait_pulse:
                self.calc_field(t)
                if np.any(np.abs(self.E) > tdse.utils.I_to_E(self.Imin)):
                    print(self.E)
                    print("Pulse reached media.")
                    self.wait_pulse = False

    def calc_field(self, t):
        self.E[:] = np.asarray(self.maxwell_ws.E)[self.N_index]

    def get_t(self):
        print((self.L - self.x0) / tdse.const.C)
        return np.arange(0, (self.L - self.x0) / tdse.const.C, self.dt)

    def calc_n(self, x):
        pass

class MaxwellNonlinearTask(Task):
    dx = tdse.utils.unit_to(20, "nm") # step in Maxwell equation
    L = tdse.utils.unit_to(200e3, "nm") # length of media
    ksi = 0.9 # maxwell propogation parameter

    x0 = tdse.utils.unit_to(60, "nm") # init location of center laser pulse

    Imin = 0.0 # Minimum intensity to start calculating the medium response

    n = None # Gas concentration
    chi = np.array([0.0, 0.0, 0.0]) # [chi^1, chi^2, chi^3, ...]

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, **kwargs)

        self.grid = tdse.grid.Grid1d(int(self.L/self.dx), self.dx)
        self.n = np.zeros(self.grid.N)
        self.x = np.linspace(0, self.L, self.n.size)

        self.wait_pulse = True

    def calc_init(self):
        self.calc_n(self.x)

        self.ws = tdse.maxwell.MaxwellWorkspace1D(self.grid)
        self.dt = self.ws.get_dt(self.ksi)

        E = self.field.E(self.field.T/2 - (self.x - self.x0)/tdse.const.C)
        self.ws.E[:] = E
        self.ws.D[:] = E

        H = self.field.E(self.field.T/2 - self.dt/2 - (self.x - self.x0 + self.grid.d/2)/tdse.const.C)
        self.ws.H[:] = H

        self.P = np.zeros(self.grid.N)

        super().calc_init()

    def calc_prop(self, i, t):
        self.P[:] = 0.0

        for i in self.chi:
            if self.chi[i] != 0.0:
                self.P[:] += self.chi[i]*self.n*self.ws.E**(i+1)

        self.ws.prop(dt=self.dt, pol=self.P)

    def get_t(self):
        return np.arange(0, (self.L - self.x0) / tdse.const.C, self.dt)

    def calc_n(self, x):
        pass
