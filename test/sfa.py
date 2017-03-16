import unittest

import tdse
import numpy as np


class SfaTest(unittest.TestCase):
    def setUp(self):
        self.I = np.logspace(13, 15, 2)
        self.E = tdse.utils.I_to_E(self.I)
        self.freq = 5.7e-2
        self.alpha = 0.1
        self.phase = np.pi/2

    def test_djdt(self):
        djdt = tdse.sfa.djdt(self.E, self.freq, self.alpha, self.phase)
