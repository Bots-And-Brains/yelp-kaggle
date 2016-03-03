import sys
from itertools import izip
import multiprocessing
import pickle
import csv
import numpy as np
from emd import emd
from datetime import datetime

from EMDMatrix import EMDMatrix

"""
This script calculates the earth-mover's distance between each pair of businesses, treating
each business as a set of points, each corresponding to the features of one of its images.

Arguments: <num-threads> <prefix for npy-format features data> <prefix for output files>
"""

size_limit = 20
chunk_size = 10
#size_limit = 40
#chunk_size = 20
#size_limit = 999999
#chunk_size = 100

class FeatureData:
	def __init__(self, file_prefix):
		self.features = np.load(file_prefix+'-features.npy')
		self.photo_ids = np.load(file_prefix+'-photo_ids.npy')
		self.business_ids = np.load(file_prefix+'-business_ids.npy')


threads = int(sys.argv[1])
features_prefix = sys.argv[2]
output_prefix = sys.argv[3]
if len(sys.argv) > 4:
	raise Exception('wrong number of args')

fdata = FeatureData(features_prefix)

features_by_bid = {}
for f, bid in izip(fdata.features, fdata.business_ids):
	if bid in features_by_bid:
		features_by_bid[bid].append(f)
	else:
		features_by_bid[bid] = [f]

for bid in features_by_bid.keys():
	features_by_bid[bid] = np.array(features_by_bid[bid])

business_ids = np.array(sorted(features_by_bid.keys())[:size_limit])

coordinates = [(i, j)
					for i in range(len(business_ids))
					for j in range(len(business_ids))
					if j < i]
chunk_coords = [coordinates[start:start+chunk_size]
					for start in range(0, len(coordinates), chunk_size)]
print "Number of chunks: %d" % len(chunk_coords)

def populate_EMD_chunk(coordinates):
	with open("out", 'a') as log:
		log.write("%s: populating (%d, %d) through (%d, %d)...\n"
			% (datetime.now().isoformat(),
			   coordinates[0][0], coordinates[0][1],
			   coordinates[-1][0], coordinates[-1][1]))
	return [(i, j, emd(features_by_bid[business_ids[i]], features_by_bid[business_ids[j]]))
				for i, j in coordinates]

if threads == 1:
	# single-threaded version -- easier debugging
	chunks = map(populate_EMD_chunk, chunk_coords)
else:
	# parallel version
	pool = multiprocessing.Pool(threads)
	chunks = pool.map(populate_EMD_chunk, chunk_coords)

emd_matrix = np.zeros((len(business_ids), len(business_ids)))
for chunk in chunks:
	for i, j, x in chunk:
		emd_matrix[i, j] = x

#	print emd_matrix

# Make it symmetrical (so far only half is populated)
emd_matrix += emd_matrix.T

EMDMatrix(business_ids, business_ids, emd_matrix).save(output_prefix)



