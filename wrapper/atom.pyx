from wavefunc cimport SWavefunc
from grid cimport SGrid
from orbitals cimport SOrbitals

def ground_state(SGrid grid):
    wf = SWavefunc(grid)
    hydrogen_ground(wf.data)
    return wf

def a_init(SGrid grid):
    orbs = SOrbitals(9, grid)
    argon_init(orbs._data)
    return orbs

def a_ort(SOrbitals orbs):
    argon_ort(orbs._data)

def n_init(SGrid grid):
    orbs = SOrbitals(5, grid)
    neon_init(orbs._data)
    return orbs

def n_ort(SOrbitals orbs):
    neon_ort(orbs._data)
