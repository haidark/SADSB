from __future__ import print_function

import h5py
import os
import numpy as np
import dicom
from scipy.misc import imresize

img_resize = True
img_shape = (128, 128)
DATA_DIR = '/media/haidar/Storage/Data/SADSB/'

def crop_resize(img):
    """
    Crop center and resize.

    :param img: image to be cropped and resized.
    """
    if img.shape[0] < img.shape[1]:
        img = img.T
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    img = crop_img
    img = imresize(img, img_shape)
    return img


def load_images(from_dir, verbose=True):
    """
    Load images in the form study x slices x width x height.
    Each image contains 30 time series frames so that it is ready for the convolutional network.

    :param from_dir: directory with images (train or validate)
    :param verbose: if true then print data
    """
    print('-'*50)
    print('Loading all DICOM images from {0}...'.format(from_dir))
    print('-'*50)

    current_study_sub = ''  # saves the current study sub_folder
    current_study = ''  # saves the current study folder
    current_study_images = []  # holds current study images
    ids = []  # keeps the ids of the studies
    study_to_images = dict()  # dictionary for studies to images
    total = 0
    images = []  # saves 30-frame-images
    from_dir = from_dir if from_dir.endswith('/') else from_dir + '/'
    for subdir, _, files in os.walk(from_dir):
        subdir = subdir.replace('\\', '/')  # windows path fix
        subdir_split = subdir.split('/')
        study_id = subdir_split[-3]
        if "sax" in subdir:
            for f in files:
                image_path = os.path.join(subdir, f)
                if not image_path.endswith('.dcm'):
                    continue

                image = dicom.read_file(image_path)
                image = image.pixel_array.astype(float)
                image /= np.max(image)  # scale to [0,1]
                if img_resize:
                    image = crop_resize(image)

                if current_study_sub != subdir:
                    x = 0
                    try:
                        while len(images) < 30:
                            images.append(images[x])
                            x += 1
                        if len(images) > 30:
                            images = images[0:30]

                    except IndexError:
                        pass
                    current_study_sub = subdir
                    current_study_images.append(images)
                    images = []

                if current_study != study_id:
                    study_to_images[current_study] = np.array(current_study_images)
                    if current_study != "":
                        ids.append(current_study)
                    current_study = study_id
                    current_study_images = []
                images.append(image)
                if verbose:
                    if total % 1000 == 0:
                        print('Images processed {0}'.format(total))
                total += 1
    x = 0
    try:
        while len(images) < 30:
            images.append(images[x])
            x += 1
        if len(images) > 30:
            images = images[0:30]
    except IndexError:
        pass

    print('-'*50)
    print('All DICOM in {0} images loaded.'.format(from_dir))
    print('-'*50)

    current_study_images.append(images)
    study_to_images[current_study] = np.array(current_study_images)
    if current_study != "":
        ids.append(current_study)

    return ids, study_to_images


def map_studies_results():
    """
    Maps studies to their respective targets.
    """
    id_to_results = dict()
    train_csv = open(DATA_DIR+'train.csv')
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, diastole, systole = item.replace('\n', '').split(',')
        id_to_results[id] = [float(diastole), float(systole)]

    return id_to_results

def append_data(ds, data):
    curRows = ds.shape[0]
    newRows = data.shape[0]
    ds.resize(curRows+newRows, axis=0)
    ds[curRows:, ...] = data


"""
Loads the training data set including X and y and saves it to .npy file.
"""
print('-'*50)
print('Writing training data to .npy file...')
print('-'*50)

study_ids, images = load_images(DATA_DIR+'train')  # load images and their ids
studies_to_results = map_studies_results()  # load the dictionary of studies to targets

study_id = study_ids[0]
x = images[study_id]
Y = []
Y.append(studies_to_results[study_id])

shpX = x.shape[1:]

with h5py.File(DATA_DIR+'trainData.h5', 'w') as dataFile:
    X = dataFile.create_dataset('X', data=x, 
                                        maxshape=(None, shpX[0], shpX[1], shpX[2]))      
    for i in range(1, len(study_ids)):
        study_id = study_ids[i]
        x = images[study_id]
        y = studies_to_results[study_id]
        for j in range(x.shape[0]):
            append_data(X, x[j,:,:,:])
            Y.append(y)
    Y = dataFile.create_dataset('Y', data=y)
  
print('Done saving Training data.')

"""
Loads the validation data set including X and study ids and saves it to .npy file.
"""
print('-'*50)
print('Writing validation data to .npy file...')
print('-'*50)

ids, images = load_images(DATA_DIR+'validate')

study_id = ids[0]
x = images[study_id]
Y = []
Y.append(study_id)

shpX = x.shape[1:]
shpY = 1
with h5py.File(DATA_DIR+'valData.h5', 'w') as dataFile:
    X = dataFile.create_dataset('X', data=x, 
                                        maxshape=(None, shpX[0], shpX[1], shpX[2]))
        
    for i in range(1, len(ids)):
        study_id = ids[i]
        x = images[study_id]
        y = study_id
        for j in range(x.shape[0]):
            append_data(X, x[j,:,:,:])
            Y.append(y)
    Y = dataFile.create_dataset('Y', data=y)

print('Done saving validation data.')
