import sys
from itertools import izip
import multiprocessing
import random
import re
import csv
from math import exp
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import EMDMatrix

"""
Train, predict, and score using SVM.
Example usage with svm trained on mean feature values:
python yelp_svm.py data/train data/train.csv -method=svm-mean -threads=2
"""

class FeatureData:
	def __init__(self, file_prefix):
		self.features = np.load(file_prefix+'-features.npy')
		self.photo_ids = np.load(file_prefix+'-photo_ids.npy')
		self.business_ids = np.load(file_prefix+'-business_ids.npy')

def load_label_data(labels_file, fdata):
	labels = np.zeros([fdata.business_ids.max()+1, 9], np.int8)
	with open(labels_file,'r') as f:
		f.readline()
		reader = csv.reader(f)
		for row in reader:
			bid = int(row[0])
			for label in [int(l) for l in row[1].split()]:
				labels[bid,label] = 1
	return labels

# Some helper functions for reading command line arguments
def parse_range(s):
	""" parse integer range from one of the following formats:
	       "123": [123]
	       "3,5,8": [3, 5, 8]
	       "3:8": [3,4,5,6,7]
	"""
	if ':' in s:
		start, end = s.split(':')
		return range(int(start), int(end))
	elif ',' in s:
		return [int(x) for x in s.split(',')]
	else:
		return [int(s)]

def parse_floats(s):
	""" parse set of floats, separated by commas (without spaces)
	"""
	if ',' in s:
		return [float(x) for x in s.split(',')]
	else:
		return [float(s)]


threads = 1
cutoff = 999999 # Set to low number to truncate set of businesses for faster testing/debugging
predict_labels = range(9) # Set of labels to predict
method = 'svm-mean'
C=1.0 # SVM smoothing parameter
gammas=[0.1] # Gamma for gaussian kernel around EMD.
             # If singleton, used on all lables; if list of same size as predict_labels, then
             # each label's model gets its own respective gamma value.
k=1 # used only by kNN
emd_matrix_prefix = None # If specified, load EMD matrix

args = []
for arg in sys.argv[1:]:
	option_match = re.match(r'(\-\w+)=(.*)', arg)
	if option_match:
		option, value = option_match.groups()
		if option == '-method':
			method = value
		elif option == '-C':
			C = float(value)
		elif option == '-gamma':
			gammas = parse_floats(value)
		elif option == '-k':
			k = int(value)
		elif option == '-emd':
			emd_matrix_prefix = value
		elif option == '-threads':
			threads = int(value)
		elif option == '-cutoff':
			cutoff = int(value)
		elif option == '-labels':
			predict_labels = parse_range(value)
		else:
			raise Exception("unknown option: %s" % option)
	else:
		args.append(arg)

features_prefix, labels_prefix = args

training = FeatureData(features_prefix)

labels = load_label_data(labels_prefix, training)

features_by_bid = {}
for f, bid in izip(training.features, training.business_ids):
	if bid in features_by_bid:
		features_by_bid[bid].append(f)
	else:
		features_by_bid[bid] = [f]

for bid in features_by_bid.keys():
	features_by_bid[bid] = np.array(features_by_bid[bid])


random.seed(12345) # ensure same psuedorandom division each time
shuffled_bids = np.unique(training.business_ids)[:cutoff]
random.shuffle(shuffled_bids)
test_set_size = shuffled_bids.size/5
shuffled_labels = labels[shuffled_bids]


# Simple, convenient smoke-test kernel. It just takes the dot product of the features of
# each business's first image. It does a terrible job.
def one_image_linear_kernel(X1, X2):
	f1 = np.array([features_by_bid[int(bid)][0] for bid in X1[:,0]])
	f2 = np.array([features_by_bid[int(bid)][0] for bid in X2[:,0]])

	return np.dot(f1, f2.T)


# Linear (dot-product) kernel that takes the average of each feature over all images 
# for each business.
def mean_linear_kernel(X1, X2):
	f1 = np.array([np.mean(features_by_bid[int(bid)].T, axis=1) for bid in X1[:,0]])
	f2 = np.array([np.mean(features_by_bid[int(bid)].T, axis=1) for bid in X2[:,0]])

	return np.dot(f1, f2.T)

def max_linear_kernel(X1, X2):
	f1 = np.array([np.ndarray.max(features_by_bid[int(bid)].T, axis=1) for bid in X1[:,0]])
	f2 = np.array([np.ndarray.max(features_by_bid[int(bid)].T, axis=1) for bid in X2[:,0]])

	return np.dot(f1, f2.T)

if emd_matrix_prefix is not None:
	emd_matrix = EMDMatrix.load(emd_matrix_prefix)

def EMD_kernel(X1, X2, gamma):
	emd = emd_matrix.for_business_ids(X1[:,0], X2[:,0])
	if emd is None:
		raise Exception('Provided EMD matrix does not cover needed business IDs.')
#	emd **= 2
	emd *= -gamma
	return np.exp(emd)

def accuracy_for_label(label_num):
	label = shuffled_labels[:,label_num]
	data = shuffled_bids.reshape(-1, 1)

	params = ''

	gamma = gammas[0] if len(gammas) == 1 else gammas[label_num]
	
	if method.startswith('svm'):
		if method == 'svm-mean':
			K = mean_linear_kernel
		elif method == 'svm-max':
			K = max_linear_kernel
		elif method == 'svm-emd':
			K = lambda X1, X2: EMD_kernel(X1, X2, gamma)
#		elif method == 'svm-gauss-mean'
	
		classifier = svm.SVC(kernel=K, C=C, verbose=False, random_state=123)

		params = 'gamma = %0.3f; ' % gamma
			
	elif method.startswith('knn'):
		weights = 'uniform' if k > 0 else 'distance'
		classifier = KNeighborsClassifier(n_neighbors=abs(k), weights=weights, n_jobs=2)

		params = 'k = %d; ' % k

	classifier.fit(data[test_set_size:], label[test_set_size:])
	pred_labels = classifier.predict(data[:test_set_size])
	accuracy = accuracy_score(label[:test_set_size], pred_labels)

	baseline = max(label[:test_set_size].mean(), 1.0 - label[:test_set_size].mean())

	print "Label %d; %sbaseline = %.3f; accuracy = %.3f" % \
		(label_num, params, baseline, accuracy)

	return {'baseline': baseline, 'accuracy': accuracy,
	        'truth': label[:test_set_size], 'prediction': pred_labels}

if threads == 1:
	# single-threaded version -- easier debugging
	results = map(accuracy_for_label, predict_labels)
else:
	# parallel version
	pool = multiprocessing.Pool(threads)
	results = pool.map(accuracy_for_label, predict_labels, chunksize=1)

for measurement in ['baseline', 'accuracy']:
	print "%s mean = %.3f" % \
		(measurement, np.array([x[measurement] for x in results]).mean())

def calculate_F1(truth_by_label, pred_by_label):
	tp = 0.0
	fn = 0.0
	fp = 0.0
	for label in range(len(truth_by_label)):
		for i, true_value in enumerate(truth_by_label[label]):
			pred_value = pred_by_label[label][i]
			if true_value == 1 and pred_value == 1:
				tp += 1.0
			elif true_value == 1 and pred_value == 0:
				fn += 1.0
			elif true_value == 0 and pred_value == 1:
				fp += 1.0
	print "tp=%.0f fn=%.0f fp=%.0f" % (tp,fn,fp)
	pre = tp / (tp + fp)
	rec = tp / (tp + fn)
	return 2*pre*rec/(pre + rec)

f1 = calculate_F1([x['truth'] for x in results], [x['prediction'] for x in results])
print "f1 = %.3f" % f1


