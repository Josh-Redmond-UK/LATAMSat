


import tensorflow as tf
import numpy as np
import rasterio as rio
import glob
import numpy as np
from scriptUtils import *


image_paths = glob.glob('latamSatData/DownloadedDataset/*/*/*.tif')

label_codes = classDictionary.keys()
label_int_rep = [i for i in range(len(label_codes))]
label_int_dict = dict(zip(label_codes, label_int_rep))

ds = tf.data.Dataset.from_tensor_slices(image_paths)

def parse_image(image_path):
    image_path_aside = image_path
    image_path = image_path.numpy().decode('utf-8')
    with rio.open(image_path) as src:
        array = src.read()
        array = np.moveaxis(array, 0, 2)

    path_parts = image_path.split('/')
    label = label_int_dict[int(path_parts[-2])]

    return array, label, image_path_aside

def convert_to_dict(img, label, path):
    return {'Image':img, 'Label':label, 'Path':path}


ds = ds.map(lambda x: tf.py_function(parse_image, [x], [tf.float32, tf.int8, tf.string]))
ds = ds.map(convert_to_dict)
