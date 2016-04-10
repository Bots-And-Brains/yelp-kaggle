from  sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import pandas


def binarize_softmax_labels(labels_matrix):
    labels_individual = dict()
    labels_shape = labels_matrix.shape
    # For every binary label (columns)..
    for i in xrange(labels_shape[1]):
        # Create a new N x 2 matrix for each label.
        tmp = np.zeros((labels_shape[0], 2), dtype="float32")
        # For every row in the label set..
        for j in xrange(labels_shape[0]):
            # Each label marks the second column as true if it's false.
            if labels_matrix[j, i] < 1:
                tmp[j] = [0., 1.]
            # Or set the first column is true.
            else:
                tmp[j] = [1., 0.]
        # Update the dict of the label.
        labels_individual[i] = tmp
    return labels_individual


def get_labels(csv_path):
    mlb = MultiLabelBinarizer()

    labels_file = os.path.join(csv_path)
    l = np.loadtxt(labels_file, dtype=np.dtype('string'), delimiter=",",skiprows=1)

    biz2labels = dict([(x[0],x[1].split()) for x in l])

    bin_labels = mlb.fit_transform(biz2labels.values())

    new_dict = dict()
    for i, bid in enumerate(biz2labels.keys()):
        new_dict[bid] = bin_labels[i]


    biz_labels_df = pandas.DataFrame.from_dict(new_dict, orient='index', dtype='bool')
    biz_labels_df.columns = ['good_for_lunch','good_for_dinner'
    ,'takes_reservations','outdoor_seating'
    ,'restaurant_is_expensive','has_alcohol'
    ,'has_table_service','ambience_is_classy'
    ,'good_for_kids']

    return biz_labels_df

def load_training_photos_from_csv(photo_to_bizid_path):
    #image_id->business_id
    imgs_file = photo_to_bizid_path
    a = np.loadtxt(imgs_file, dtype=np.dtype('string'), delimiter=",",skiprows=1)
    img2biz = dict(a)
    return img2biz


# Takes a matrix of binary labels, and outputs the label indexes,
# or labels from label_map if provided.
def matrix_2_labels(x, label_map=None):
    x_len, x_width = x.shape
    out = [None] * x_len
    for i in range(x_len):
        out[i] = list()
        for label in range(x_width):
            #print x[i][0][0]
            if x.item((i, label)) > 0:
                if label_map == None:
                    out[i].append(label)
                else:
                    out[i].append(label_map[label])

    return out
