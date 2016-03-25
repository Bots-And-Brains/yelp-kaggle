from  sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

labels_file = os.path.join(os.curdir,"../data/train.csv")
l = np.loadtxt(labels_file, dtype=np.dtype('string'), delimiter=",",skiprows=1)

biz2labels = dict([(x[0],x[1].split()) for x in l])

bin_labels = mlb.fit_transform(biz2labels.values())

new_dict = dict()
for i, bid in enumerate(biz2labels.keys()):
    new_dict[bid] = bin_labels[i]

#biz2labels = dict((biz2labels.keys(), bin_labels))

biz_labels_df = pandas.DataFrame.from_dict(new_dict, orient='index', dtype='bool')
biz_labels_df.columns = ['good_for_lunch','good_for_dinner'
,'takes_reservations','outdoor_seating'
,'restaurant_is_expensive','has_alcohol'
,'has_table_service','ambience_is_classy'
,'good_for_kids']
