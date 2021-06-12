import importlib
import numpy as np


def load(name):
    module = importlib.import_module(name)
    task = module.Task()
    task.load()
    return task

def get_index_aw(task, N: int) -> int:
    return int(N / (np.pi/task.dt/task.freq) * task.aw.size)
