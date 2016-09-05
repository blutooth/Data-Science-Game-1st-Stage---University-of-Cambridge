from PIL import Image, ImageChops, ImageOps
import numpy as np
import glob
import os
import csv
from utils import *


def read_labels(csv_file):
    d = {}
    reader = csv.reader(open(csv_file, "rb"))
    for rows in reader:
        if rows[1] == 'label':
            continue
        d[rows[0]] =  int(rows[1])
    return d

#http://stackoverflow.com/questions/9103257/resize-image-maintaining-aspect-ratio-and-making-portrait-and-landscape-images-e
def makeThumb(f_in, size=(96,96), pad=True):
    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    if pad:
        thumb = image.crop( (0, 0, size[0], size[1]) )

        offset_x = max( (size[0] - image_size[0]) / 2, 0 )
        offset_y = max( (size[1] - image_size[1]) / 2, 0 )

        thumb = ImageChops.offset(thumb, offset_x, offset_y)

    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))

    return thumb

labels = read_labels('id_train.csv')
img_list = sorted(glob.glob(os.path.join(DATA_DIR, 'roof_images') + '/*.jpg'))


X = []
y = []
X_no_pad = []
X_test = []
X_test_no_pad = []
X_no_label = []
test_ids = []

test_set = read_labels('sample_submission4.csv')

for img_f in img_list:
    img_id = os.path.basename(img_f).split('.')[0]
    img = makeThumb(img_f)

    if img_id in labels:
        X.append(np.array(img))
        y.append(labels[img_id])
        img_no_pad = makeThumb(img_f, size=(224,224), pad=False)
        X_no_pad.append(np.array(img_no_pad))
    elif img_id in test_set:
        X_test.append(np.array(img))
        test_ids.append(img_id)
        img_no_pad = makeThumb(img_f, size=(224,224), pad=False)
        X_test_no_pad.append(np.array(img_no_pad))
    else:
        X_no_label.append(np.array(img))

X = np.array(X)
y = np.array(y)
X_test = np.array(X_test)
np.save(os.path.join(DATA_DIR,'X_train.npy'), X)
np.save(os.path.join(DATA_DIR,'y_train.npy'), y)
np.save(os.path.join(DATA_DIR,'X_test.npy'), X_test)
np.save(os.path.join(DATA_DIR,'test_ids.npy'), test_ids)

X_no_label = np.array(X_no_label)
np.save(os.path.join(DATA_DIR,'X_no_label.npy'), X_no_label)


np.save(os.path.join(DATA_DIR,'X_train_224.npy'), X_no_pad)
np.save(os.path.join(DATA_DIR,'X_test_224.npy'), X_test_no_pad)
