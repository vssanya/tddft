from wavefunc cimport SWavefunc
from grid cimport SGrid

def ground_state(SGrid grid):
    wf = SWavefunc(grid)
    hydrogen_ground(wf.data)
    return wf
