import numpy as np

import tdse


def test_rank_equal_dist():
    rank = tdse.utils.rank_equal_dist(11, 3)
    print(rank)
    assert rank.size == 11
    assert np.all(rank[0:4] == 0) and np.all(rank[4:4+4] == 1) and np.all(rank[8:] == 2)
