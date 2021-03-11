import time
from .task import CalcData, TimeShapeMixin


class CalcTimeStepBenchmarkData(TimeShapeMixin, CalcData):
    NAME = "calc_dt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.start_time = time.time()

    def calc(self, task, i, t):
        current_time = time.time()
        self.dset[i] = current_time - self.start_time
        self.start_time = current_time
