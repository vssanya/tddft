import time
import numpy as np

import tdse


# class OrbitalsFunc:
    # timer = time.time

    # def setup(self):
        # dr = 0.02
        # r_max = 200
        # Nr = int(r_max/dr)
        # Nl = 60

        # self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        # self.atom = tdse.atom.Ar
        # self.orbs = tdse.orbitals.Orbitals(self.atom, self.grid)
        # self.orbs.init()
        # self.field = tdse.field.TwoColorSinField()

    # def time_norm(self):
        # self.orbs.norm()

    # def time_orbs_az(self):
        # tdse.calc.az(self.orbs, self.atom, self.field, 0.0)

    # def time_ionization_prob(self):
        # tdse.calc.ionization_prob(self.orbs)

class OrbitalsPropagate:
    timer = time.time

    def setup(self):
        dr = 0.02
        r_max = 200
        Nr = 10000
        Nl = 512

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        self.sp_grid = tdse.grid.SpGrid(Nr, 33, 1, r_max)
        self.n = np.ndarray((33, Nr))

        self.ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, self.sp_grid)

        self.atom = tdse.atom.Ar
        self.orbs = tdse.orbitals.Orbitals(self.atom, self.grid)
        N = self.atom.countOrbs
        self.orbs.asarray()[:] = np.random.random((N, Nl, Nr)) + 1j*np.random.random((N, Nl, Nr))
        self.orbs.init()
        self.uh = np.ndarray(Nr)
        self.uabs = tdse.abs_pot.UabsCache(tdse.abs_pot.UabsMultiHump(0.1, 10), self.grid)
        self.ws = tdse.workspace.SOrbsWorkspace(tdse.atom.ShAtomCache(self.atom, self.grid), self.grid, self.sp_grid, self.uabs, self.ylm_cache, 1, 3)
        self.field = tdse.field.TwoColorSinField()

    def time_hartree_potential_l0(self):
        tdse.hartree_potential.potential(self.orbs, 0, self.uh)

    def time_hartree_potential_l1(self):
        tdse.hartree_potential.potential(self.orbs, 1, self.uh)

    def time_hartree_potential_l2(self):
        tdse.hartree_potential.potential(self.orbs, 2, self.uh)

    def time_uxc_l0(self):
        tdse.hartree_potential.UXC_LB.calc_l0(0, self.orbs, self.sp_grid, self.ylm_cache, self.uh, self.n)

    def time_n_l0(self):
        self.orbs.n_l0(self.n[0])

    # def time_lda(self):
        # tdse.hartree_potential.lda(0, self.orbs, self.sp_grid, self.ylm_cache, self.uh)

    def time_orbitals_propagate(self):
        self.ws.prop(self.orbs, self.field, 0.0, 0.1)

    def time_calc_uee(self):
        self.ws.calc_uee(self.orbs)


class Wf:
    timer = time.time

    def setup(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 512*2

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        # self.sp_grid = tdse.grid.SpGrid(Nr, 32, 1, r_max)
        # self.ylm_cache = tdse.sphere_harmonics.YlmCache(Nl, self.sp_grid)
        self.atom = tdse.atom.H
        self.atom_cache = tdse.atom.ShAtomCache(self.atom, self.grid)
        # self.n = np.ndarray((Nr, 32))
        self.wf = tdse.wavefunc.ShWavefunc(self.grid)
        self.wf.asarray()[:] = np.random.random((Nl, Nr)) + 1j*np.random.random((Nl, Nr))
        self.uabs = tdse.abs_pot.UabsMultiHump(0.1, 10)
        self.ws = tdse.workspace.ShWavefuncWS(self.atom_cache, self.grid, tdse.abs_pot.UabsCache(self.uabs, self.grid))
        self.field = tdse.field.TwoColorSinField()

    def time_prop(self):
        for i in range(10):
            self.ws.prop(self.wf, self.field, 0.0, 0.1)

    def time_prop_abs(self):
        for i in range(100):
            self.ws.prop_abs(self.wf, 0.1)

    # def time_z(self):
        # self.wf.z()

    def time_calc_az(self):
        tdse.calc.az(self.wf, self.atom_cache, self.field, 0.0)

    # def time_n_sp(self):
        # self.wf.n_sp(self.sp_grid, self.ylm_cache, self.n)

    # def time_norm(self):
        # self.wf.norm()

class WfGPU:
    timer = time.time

    def setup(self):
        dr = 0.02
        r_max = 200
        Nr = int(r_max/dr)
        Nl = 512*2

        self.grid = tdse.grid.ShGrid(Nr, Nl, r_max)
        self.atom = tdse.atom.H
        self.atom_cache = tdse.atom.ShAtomCache(self.atom, self.grid)

        self.wf = tdse.wavefunc.ShWavefunc(self.grid)
        self.wf.asarray()[:] = np.random.random((Nl, Nr)) + 1j*np.random.random((Nl, Nr))

        self.wf_device = tdse.wavefunc_gpu.ShWavefuncGPU(self.wf)

        self.uabs = tdse.abs_pot.UabsMultiHump(0.1, 10)
        self.ws = tdse.workspace_gpu.WfGPUWorkspace(self.atom_cache, self.grid, tdse.abs_pot.UabsCache(self.uabs, self.grid), threadsPerBlock=8)
        self.field = tdse.field.TwoColorSinField()

    def time_prop(self):
        for i in range(10):
            self.ws.prop(self.wf_device, self.field, 0.0, 0.1)
        wf = self.wf_device.get()

    def time_prop_abs(self):
        for i in range(100):
            self.ws.prop_abs(self.wf_device, 0.1)
        wf = self.wf_device.get()

    def time_copy_wf(self):
        wf = self.wf_device.get()

    def time_calc_az(self):
        tdse.calc.az(self.wf_device, self.atom_cache, self.field, 0)
