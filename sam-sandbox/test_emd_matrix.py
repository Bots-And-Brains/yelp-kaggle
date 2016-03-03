import sys
from itertools import izip
import multiprocessing
import pickle
import csv
import numpy as np
from emd import emd
from datetime import datetime

import EMDMatrix

class FeatureData:
	def __init__(self, file_prefix):
		self.features = np.load(file_prefix+'-features.npy')
		self.photo_ids = np.load(file_prefix+'-photo_ids.npy')
		self.business_ids = np.load(file_prefix+'-business_ids.npy')


features_prefix = sys.argv[1]
fdata = FeatureData(features_prefix)

features_by_bid = {}
for f, bid in izip(fdata.features, fdata.business_ids):
	if bid in features_by_bid:
		features_by_bid[bid].append(f)
	else:
		features_by_bid[bid] = [f]

for bid in features_by_bid.keys():
	features_by_bid[bid] = np.array(features_by_bid[bid])

business_ids = np.array(sorted(features_by_bid.keys())[:2])
print "bids: ", business_ids

print "recalculated:"

for bid1 in business_ids:
	for bid2 in business_ids:
		print "D(%d, %d): %.4f" % (bid1, bid2,
		                           emd(features_by_bid[bid1], features_by_bid[bid2]))

print "from file:"

emd_matrix = EMDMatrix.load(sys.argv[2])
print emd_matrix.for_business_ids(business_ids, business_ids)



