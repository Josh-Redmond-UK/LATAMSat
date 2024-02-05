
import numpy as np
import tensorflow as tf
import glob

classDictionary = {30: 'Herbaceous Vegetation',
20: 'Shrubs',
40: 'Agricultural Land',
50: 'Urban Areas',
60: 'Bare Earth and Sparse Vegetation',
70: 'Snow and Ice',
80: 'Permanent Water Bodies',
90: 'Herbaceous Wetland',
100: 'Moss and Lichen',
111: 'Closed Evergreen Needle Leaf Forest',
112: 'Closed Evergreen Broad Leaf Forest',
113: 'Closed Deciduous Needle Leaft Forest',
114: 'Closed Deciduous Broad Leaf Forest',
115: 'Closed Mixed Forest',
116: 'Other Closed Forest',
121: 'Open Evergreen Needle Leaf Forest',
122: 'Open Evergreen Broad Leaf Forest',
123: 'Open Deciduous Needle Leaft Forest',
124: 'Open Deciduous Broad Leaf Forest',
125: 'Open Mixed Forest',
126: 'Other Open Forest',
200: 'Oceans, Seas'}

encoding_dictionary = dict(zip(classDictionary.values(), [i for i in range(len(classDictionary.keys()))]))


def get_lulc_class(path):
    splits = path.split('/')
    return encoding_dictionary[splits[-2]]#tf.one_hot(encoding_dictionary[splits[-2]], len(encoding_dictionary.keys()))

def multispectral_to_rgb_linear(raster, optical_maximum = 2000):

    r = raster[:, :, 3]
    g = raster[:, :, 2]
    b = raster[:, :, 1]

    rgb_raster = np.stack([r, g, b], axis=2)

    #cast to uint and scale to 0/255

    rgb_raster = rgb_raster/optical_maximum
    rgb_raster = np.around(rgb_raster*255)
    rgb_raster = np.clip(rgb_raster, 0, 255).astype(int)
    return rgb_raster



def get_paths(top_level_path):
    file_pattern = top_level_path + '*/*/*.tif'
    file_paths = glob.glob(file_pattern)
    return file_paths



def get_tiff_dataset_from_folder(top_level_path):
    file_pattern = top_level_path + '/*/*.tif'
    file_paths = glob.glob(file_pattern)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_tiff_image)

    return dataset

def load_tiff_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_tiff(image, index=0, name=None)
    image = tf.clip_by_value(tf.cast(image, tf.float32) / tf.maximum(image), 0, 1)  # Normalize to [0, 1]

    return image


def get_dataset(file_pattern, mode="rgb", shuffle_size="full"):
    file_paths = glob.glob(file_pattern)
    labels = list(map(get_lulc_class, file_paths))


    dataset = tf.data.Dataset.from_tensor_slices((file_paths,labels))
    if shuffle_size == "full":
        shuffle_size = dataset.cardinality()

    if mode == "rgb":
        dataset = dataset.map(load_rgb_image).shuffle(buffer_size=shuffle_size).batch(32)
    else:
        dataset = dataset.map(load_tiff_image).shuffle(buffer_size=shuffle_size).batch(32)

    
    return dataset

def load_rgb_image(file_path, label):
    #file_path, label = tensor
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([64, 64, 3])
    image = tf.cast(image, tf.float32) / 255 # Normalize to [0, 1]
    return image, label

