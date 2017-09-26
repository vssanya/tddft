import numpy as np

import tdse
import tdse.task


class ExampleTask(tdse.task.WavefuncTask):
    atom = tdse.atom.H

    dt = 0.025
    dr = 0.125
    r_max = 60
    Nl = 8

    field = tdse.field.TwoColorSinField()
    nT = 0

    uabs = tdse.abs_pot.UabsMultiHump(2, 25)

    CALC_DATA = ['az']

    def calc_data(self, i, t):
        self.az[i] = tdse.calc.az(self.wf, self.atom, self.field, t)

    def calc_init(self):
        super().calc_init()

        self.az = np.zeros(self.t.shape)

if __name__ == '__main__':
    task = ExampleTask()
    task.calc()
