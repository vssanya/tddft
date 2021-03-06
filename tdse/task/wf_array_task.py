import os
import numpy as np

import tdse
from .task import CalcData, TaskAtom


class TimeShapeMixin(object):
    def get_shape(self, task):
        return (task.t.size, task.N)

class AzWfArrayData(TimeShapeMixin, CalcData):
    NAME = "az"

    def calc_init(self, task, file):
        super().calc_init(task, file)
        self.az = np.zeros(self.dset.shape[1])

    def calc(self, task, i, t):
        task.calc_field(t)
        tdse.calc_gpu.az(task.wf_array, task.atom_cache, task.E, self.az)

        self.dset[i] = self.az

class WavefuncArrayTask(TaskAtom):
    is_calc_ground_state = True

    N = 10

    def __init__(self, path_res='res', mode=None, **kwargs):
        self.use_gpu = kwargs.pop("use_gpu", False)
        assert(not (self.use_gpu and kwargs.get("is_mpi", False)))
        if self.use_gpu:
            gpu_device_id = int(os.environ.get("CUDA_DEVICE", "0"))
            tdse.calc.setGpuDevice(gpu_device_id)

            print("Set gpu device id = {}".format(gpu_device_id))

        super().__init__(path_res, mode, **kwargs)

    def save_state(self, i):
        np.save(os.path.join(self.save_path, 'wf_{}.npy'.format(i)), self.wf.asarray())

    def load_state(self, i):
        self.wf.asarray[:] = np.load(os.path.join(self.save_path, 'wf_{}.npy'.format(i)))

    def _get_ws_class(self):
        if self.use_gpu:
            return tdse.workspace_gpu.WfArrayGPUWorkspace
        else:
            return tdse.workspace.ShWfArrayWS

    def _create_wf_array(self, wf_gs):
        if self.use_gpu:
            wf_array = tdse.wavefunc_gpu.ShWavefuncArrayGPU(wf_gs, self.N)
        else:
            m = np.full(self.N, 0, dtype=np.intc)
            rank = None
            if self.is_mpi:
                rank = tdse.utils.rank_equal_dist(self.N, self.size)
            wf_array = tdse.wavefunc.ShWavefuncArray(self.N, m, wf_gs.grid, self.comm, rank)

        return wf_array

    def calc_init(self):
        super().calc_init()

        self.wf_gs = self.calc_ground_state()
        self.wf_array = self._create_wf_array(self.wf_gs)

        self.ws = self._get_ws_class()(self.atom_cache, self.sh_grid, self.uabs_cache, self.N)

        if self.is_root:
            self.E = np.zeros(self.N)
        else:
            self.E = None

    def calc_ground_state(self):
        ws = tdse.workspace.ShWavefuncWS(self.atom_cache, self.sh_grid, self.uabs_cache)
        return tdse.ground_state.wf(self.atom, self.sh_grid, ws, self.dt, 10000)[0]

    def calc_prop(self, i, t):
        if self.is_root:
            self.calc_field(t + self.dt/2)

        if self.is_mpi:
            self.comm.Barrier()

        self.ws.prop(self.wf_array, self.E, self.dt)

    def calc_field(self, t):
        pass
