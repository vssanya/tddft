import tdse
import tdse.maxwell

import numpy as np

from .task import Task
from .wf_array_task import CalcData, WavefuncArrayTask


class EdepsTData(CalcData):
    NAME = "Et"

    def __init__(self, x, **kwargs):
        super().__init__(**kwargs)
        
        self.x = x

    def get_shape(self, task):
        if issubclass(type(task), MaxwellTDSETask):
            return (task.t.size//task.m_dt,)
        else:
            return (task.t.size,)

    def calc_init(self, task, file):
        super().calc_init(task, file)

        if issubclass(type(task), MaxwellTDSETask):
            self.x_index = int(self.x / task.maxwell_full_grid.d)
        else:
            self.x_index = int(self.x / task.grid.d)

    def calc(self, task, i, t):
        if task.is_root:
            if issubclass(type(task), MaxwellTDSETask):
                if i % task.m_dt == 0 and i // task.m_dt < self.dset.size:
                    if self.x_index < task.maxwell_ws.E.size + task.window_N_start and self.x_index >= task.window_N_start:
                        self.dset[i // task.m_dt] = task.maxwell_ws.E[self.x_index - task.window_N_start]
                    else:
                        self.dset[i // task.m_dt] = 0.0
            else:
                self.dset[i] = task.ws.E[self.x_index]


class EdepsXData(CalcData):
    NAME = "Ex"

    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        
        self.t = t

    def get_shape(self, task):
        if issubclass(type(task), MaxwellTDSETask):
            return (self.t.size, task.maxwell_window_grid.N)
        else:
            return (self.t.size, task.grid.N)

    def calc_init(self, task, file):
        super().calc_init(task, file)

        self.t_index = (self.t / task.dt).astype(int)

    def calc(self, task, i, t):
        if task.is_root:
            if np.any(self.t_index == i):
                print("Data: ", np.argwhere(self.t_index == i))
                if issubclass(type(task), MaxwellTDSETask):
                    self.dset[np.argwhere(self.t_index == i),:] = task.maxwell_ws.E[:]
                else:
                    self.dset[np.argwhere(self.t_index == i),:] = task.ws.E[:]

class MaxwellTDSETask(WavefuncArrayTask):
    dx = tdse.utils.unit_to(20, "nm") # step in Maxwell equation
    L = tdse.utils.unit_to(200e3, "nm") # length of media
    ksi = 0.9 # maxwell propogation parameter

    use_move_window = False
    calc_tdse = True

    x0 = tdse.utils.unit_to(60, "nm") # init location of center laser pulse

    Lw = None

    Imin = 0.0 # Minimum intensity to start calculating the medium response

    n = None # Gas concentration
    chi = None # np.array([chi^1, chi^2, chi^3, ...])

    def __init__(self, path_res='res', mode=None, **kwargs):
        super().__init__(path_res, mode, **kwargs)

        if self.use_move_window:
            if self.Lw is None:
                self.Lw = self.field.T * tdse.const.C
            self.Ndt_move = int(0.05 * self.Lw / (tdse.const.C*self.dt))
            self.window_N_start = int((self.x0 - self.Lw / 2)/self.dx)
            self.window_P_index = 0
        else:
            self.Lw = self.L
            self.window_N_start = 0
            self.window_P_index = 0

        self.maxwell_full_grid = tdse.grid.Grid1d(int(self.L/self.dx), self.dx)
        self.maxwell_window_grid = tdse.grid.Grid1d(int(self.Lw/self.dx), self.dx)

        if self.is_root:
            print("Maxwell grid: N = {}, dx = {}, is_move = {}".format(
                self.maxwell_window_grid.N, tdse.utils.to_nm(self.dx),
                self.use_move_window))

            print("Size of window Lw = {} nm".format(tdse.utils.to_nm(self.Lw)))

        self.n = np.zeros(self.maxwell_full_grid.N)
        self.x = np.linspace(0, self.L, self.n.size)

        self.wait_pulse = True

    def calc_init(self):
        self.calc_n(self.x)

        self.N_index = np.argwhere(self.n != 0.0).reshape(-1)
        if self.use_move_window:
            self.N = self.maxwell_window_grid.N
        else:
            self.N = self.N_index.size
            
        if self.is_root:
            print("Count tdse equations N = {}".format(self.N))

        if self.is_root:
            self.az = np.zeros((2,self.N))
            self.vz = np.zeros((2,self.N))
            self.z  = np.zeros((2,self.N))

            self.maxwell_ws = tdse.maxwell.MaxwellWorkspace1D(self.maxwell_window_grid)
            self.m_dt = int(self.maxwell_ws.get_dt(self.ksi) / self.dt)
            print("Maxwell time step dt = {} fs".format(tdse.utils.to_fs(self.m_dt*self.dt)))

            E = self.field.E(self.field.T/2 - (self.x_window - self.x0)/tdse.const.C)
            self.maxwell_ws.E[:] = E
            self.maxwell_ws.D[:] = E

            H = self.field.E(self.field.T/2 - self.maxwell_ws.get_dt(self.ksi)/2 -
                    (self.x_window - self.x0 + self.maxwell_window_grid.d/2)/tdse.const.C)
            self.maxwell_ws.H[:] = H

            self.P = np.zeros(self.maxwell_window_grid.N)
        else:
            self.az = [None, None]
            self.m_dt = None

        if self.is_mpi:
            self.m_dt = self.comm.bcast(self.m_dt, root=0)

        super().calc_init()

    def calc_prop(self, i, t):
        if not self.wait_pulse and self.calc_tdse:
            super().calc_prop(i, t)

            if self.is_root:
                self.az[0] = self.az[1]
                self.vz[0] = self.vz[1]
                self.z[0] = self.z[1]

            if self.use_gpu:
                tdse.calc_gpu.az(self.wf_array, self.atom_cache, self.E, self.az[1])
            else:
                tdse.calc.az_array(self.wf_array, self.atom_cache, self.E, self.az[1])

            if self.is_root:
                self.vz[1] = self.vz[0] + (self.az[0] + self.az[1])*self.dt/2
                self.z[1]  = self.z[0]  + (self.vz[0] + self.vz[1])*self.dt/2

        if i % self.m_dt == 0:
            if self.is_root:
                if self.use_move_window:
                    self.P[:] = - self.z[1][self.from_wf_index]*self.n_window
                else:
                    self.P[self.N_index] = - self.z[1]*self.n[self.N_index]

                if self.chi is not None:
                    for i in range(self.chi.size):
                        if self.chi[i] != 0.0:
                            self.P[:] += self.chi[i]*self.n_window*self.maxwell_ws.E**(i+1)

                self.maxwell_ws.prop(dt=self.dt*self.m_dt, pol=self.P)

            if self.use_move_window and i % self.Ndt_move:
                if self.is_root:
                    N_shift = self.maxwell_ws.move_center_window_to_max_E()
                else:
                    N_shift = None

                if self.is_mpi:
                    N_shift = self.comm.bcast(N_shift, root=0)

                self.window_N_start += N_shift

                self.window_P_index += N_shift
                self.window_P_index = self.window_P_index % self.maxwell_window_grid.N

                if not self.wait_pulse:
                    for i in range(N_shift):
                        self.wf_array.set(self.window_P_index-i-1, self.wf_gs)

                    if self.is_root:
                        self.az[1][self.window_P_index-N_shift:self.window_P_index] = 0.0
                        self.vz[1][self.window_P_index-N_shift:self.window_P_index] = 0.0
                        self.z[1][self.window_P_index-N_shift:self.window_P_index] = 0.0

            if self.wait_pulse:
                if self.use_move_window:
                    if np.any(self.n_window > 0.0):
                        self.wait_pulse = False
                else:
                    if self.is_root:
                        self.calc_field(t)
                        if np.any(np.abs(self.E) > tdse.utils.I_to_E(self.Imin)):
                            self.wait_pulse = False

                    if self.is_mpi:
                        self.wait_pulse = self.comm.bcast(self.wait_pulse, root=0)

                if not self.wait_pulse and self.is_root:
                    print("Pulse reached media.")

        if self.is_mpi:
            self.comm.Barrier()

    @property
    def to_wf_index(self):
        return np.r_[self.maxwell_window_grid.N-self.window_P_index:self.maxwell_window_grid.N,0:self.maxwell_window_grid.N-self.window_P_index]

    @property
    def from_wf_index(self):
        return np.r_[self.window_P_index:self.maxwell_window_grid.N,0:self.window_P_index]

    @property
    def n_window(self):
        return self.n[self.window_N_start:self.window_N_start+self.maxwell_window_grid.N]

    @property
    def x_window(self):
        return self.x[self.window_N_start:self.window_N_start+self.maxwell_window_grid.N]

    def calc_field(self, t):
        if self.use_move_window:
            self.E[:] = self.maxwell_ws.E[self.to_wf_index]
        else:
            self.E[:] = self.maxwell_ws.E[self.N_index]

    def get_t(self):
        if self.use_move_window:
            return np.arange(0, (self.L - self.x0 - self.Lw/2) / tdse.const.C, self.dt)
        else:
            return np.arange(0, (self.L - self.x0) / tdse.const.C - self.field.T/2, self.dt)

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

        for i in range(self.chi.size):
            if self.chi[i] != 0.0:
                self.P[:] += self.chi[i]*self.n*self.ws.E**(i+1)

        self.ws.prop(dt=self.dt, pol=self.P)

    def get_t(self):
        return np.arange(0, (self.L - self.x0) / tdse.const.C, self.dt)

    def calc_n(self, x):
        pass
