import sys
from itertools import izip
import numpy as np


"""
Output a toy data set to make it easier to verify correctness of EMD.
"""

photo_ids = np.array(range(5))
bids = np.array([0, 0, 1, 1, 2])
features = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1],
                     [0, 0.5]])
np.save('data/debug-photo_ids', photo_ids)
np.save('data/debug-business_ids', bids)
np.save('data/debug-features', features)


