import numpy as np
import importlib
import sys

def get_count_mpi_task(task):
    if task.active_orbs is None:
        return task.atom.countOrbs
    else:
        return np.sum(task.active_orbs)

if __name__ == "__main__":
    module = importlib.import_module(sys.argv[1])
    task = module.Task(task_id=0, calc_rmax=False)
    #task = module.Task(calc_rmax=False)
    print("Count mpi tasks = {}".format(get_count_mpi_task(task)))
