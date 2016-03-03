import sys
from itertools import izip
import multiprocessing
import pickle
import csv
import numpy as np

"""
Quick script to convert features.pickle to .npy format.
(I was having trouble with the pickle file in the python version of my EC2 instance.)
"""

with open(sys.argv[1],'rb') as f:
	all_data = pickle.load(f)

features = all_data.as_matrix()[:,2:]
photo_ids = all_data['photo_id']
business_ids = all_data['business_id']

np.save('data/train-features', features)
np.save('data/train-photo_ids', photo_ids)
np.save('data/train-business_ids', business_ids)

