import numpy as np
import pickle


def load_full(pickle_file):

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)

        out = dict()

        out['train_dataset'] = np.matrix(save['train_data'], dtype='float32')
        out['train_labels'] = np.matrix(save['train_labels'], dtype='float32')
        out['valid_dataset'] = np.matrix(save['validate_data'], dtype='float32')
        out['valid_labels'] = np.matrix(save['validate_labels'], dtype='float32')
        out['test_dataset'] = np.matrix(save['test_data'], dtype='float32')
        out['test_bids'] = list(save['test_business_ids'])


        del save  # hint to help gc free up memory
        print 'Training set', out['train_dataset'].shape, out['train_labels'].shape
        print 'Validation set', out['valid_dataset'].shape, out['valid_labels'].shape
        print 'Test set', out['test_dataset'].shape, len(out['test_bids'])

        return out


#def save_full(data, pickle_file):
#     with open(pickle_file, 'rb') as f:
#          save = pickle.load(f)



# Takes the  same input as the caffe features extractor, and also the output.
# Returns a list of features
def process_photo_features(caffe_features_input_file, caffe_features_output_file, limit=None):

    from pathlib2 import PosixPath

    features = np.loadtxt(caffe_features_output_file, dtype=np.dtype('float64'), delimiter=" ",skiprows=0)

    image_paths= open(caffe_features_input_file, 'r')

    image_features = dict()

    for i in range(len(features)):
        img = PosixPath(image_paths.readline())
        i_period = img.name.find('.')
        image_features[int(img.name[:i_period])] = features[i]

    image_paths.close()

    return image_features