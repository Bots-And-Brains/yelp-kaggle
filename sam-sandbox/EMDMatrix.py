import sys
from itertools import izip
import pickle
import numpy as np


class EMDMatrix:
	"""
	Wrapper for earth-movers distance matrix created by build_emd_matrix.py.
	Automatically re-indexes for specified business IDs.
	"""
	def __init__(self, row_business_ids, col_business_ids, D):
		self.row_business_ids = row_business_ids
		self.col_business_ids = col_business_ids
		self.D = D

		self.row_index_by_bid = {}
		for i, bid in enumerate(row_business_ids):
			self.row_index_by_bid[bid] = i
		self.col_index_by_bid = {}
		for i, bid in enumerate(col_business_ids):
			self.col_index_by_bid[bid] = i

	def for_business_ids(self, row_ids, col_ids):
		try:
			row_indexes = np.array([self.row_index_by_bid[bid] for bid in row_ids])
			col_indexes = np.array([self.col_index_by_bid[bid] for bid in col_ids])
			return self.D[row_indexes][:,col_indexes]
		except KeyError:
			return None

	def save(self, file_prefix):
		np.save(file_prefix+'-rowids', self.row_business_ids)
		np.save(file_prefix+'-colids', self.col_business_ids)
		np.save(file_prefix+'-data', self.D)


def load(file_prefix):
	return EMDMatrix(np.load(file_prefix+'-rowids.npy'),
	                 np.load(file_prefix+'-colids.npy'),
	                 np.load(file_prefix+'-data.npy'))


def test():
	row_ids = np.array([2,3,4])
	col_ids = np.array([3,4,5])
	D = np.outer(row_ids, col_ids)
	print D

	M = EMDMatrix(row_ids, col_ids, D)

	D_reindexed = M.for_business_ids(np.array([3,4]), np.array([4,5,3]))
	print D_reindexed

	D_bogus = M.for_business_ids(np.array([3,4,5]), np.array([4,5,3]))
	print D_bogus


if __name__ == '__main__':
	test()


