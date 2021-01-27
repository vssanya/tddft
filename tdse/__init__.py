import os
from ctypes.util import find_library
WITH_GPU = False#os.environ.get("TDSE_WITH_GPU") is not None or find_library('cuda') is not None

from . import utils

from . import grid
from . import masks

from . import carray

from . import sphere_harmonics
from . import abs_pot
from . import wavefunc

from . import atom
if WITH_GPU:
    from . import wavefunc_gpu

from . import orbitals
from . import field
from . import hartree_potential

from . import workspace
if WITH_GPU:
    from . import workspace_gpu

from . import calc
if WITH_GPU:
    from . import calc_gpu

from . import ground_state
from . import tdsfm

from . import sfa

from . import maxwell

GAUGE_LENGTH = 0
GAUGE_VELOCITY = 1
