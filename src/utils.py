
import numpy as np
import tensorflow as tf
import sklearn.model_selection
import random
import glob
import matplotlib.pyplot as plt
import tiffile as tiff

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
encoding_dictionary_ms = dict(zip(classDictionary.keys(), [i for i in range(len(classDictionary.keys()))]))
reverse_encoding_dictionary = dict(zip(list(encoding_dictionary.values()), list(encoding_dictionary.keys())))

print(encoding_dictionary)

def get_lulc_class(path):
    splits = path.split('/')
    try:
        return encoding_dictionary[splits[-2]]#tf.one_hot(encoding_dictionary[splits[-2]], len(encoding_dictionary.keys()))
    except:
        return encoding_dictionary_ms[int(splits[-2])]#tf.one_hot(encoding_dictionary[splits[-2]], len(encoding_dictionary.keys()))


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

def load_tiff_image(file_path, label):
    print("loading tiff")
    image = tiff.imread(file_path)
    print("converting to tensor")
    image = tf.convert_to_tensor(image)
    #image.set_shape([64, 64, 12])
    print("rescaling")
    image = tf.clip_by_value(tf.cast(image, tf.float32) / tf.maximum(image), 0, 1)  # Normalize to [0, 1]

    return image, label


def get_dataset(file_pattern, mode="rgb", shuffle=True, test_size=0.2):
    file_paths = glob.glob(file_pattern)

    if shuffle:
        random.shuffle(file_paths)


    train_paths, test_paths = sklearn.model_selection.train_test_split(file_paths, test_size=test_size, shuffle=False)

    train_labels = list(map(get_lulc_class, train_paths))
    test_labels = list(map(get_lulc_class, test_paths))
    print("Verifying paths and labels")
    print(train_paths[:15])
    print(train_labels[:15])
    print(train_paths[-15:])
    print(train_labels[-15:])

    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths,train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths,test_labels))


#    if shuffle_size == "full":
#        shuffle_size = dataset.cardinality()

    if mode == "rgb":
        train_dataset = train_dataset.map(load_rgb_image).batch(32)
        test_dataset = test_dataset.map(load_rgb_image).batch(32)

    else:
        train_dataset = train_dataset.map(lambda x, y: tf.py_function(load_tiff_image, [x,y], [tf.float32, tf.string])).batch(32)
        test_dataset = test_dataset.map(lambda x, y: tf.py_function(load_tiff_image, [x,y], [tf.float32, tf.string])).batch(32)

    
    return train_dataset, test_dataset

def load_rgb_image(file_path, label):
    print(file_path, label)    
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([64, 64, 3])
    image = tf.cast(image, tf.float32) / 255 # Normalize to [0, 1]
    return image, label

def show_batch(batch):
  plt.figure(figsize=(16, 16))
  for n in range(min(32, 16)):
      ax = plt.subplot(32//8, 8, n + 1)
      # show the image
      plt.imshow(batch[0][n])
      # and put the corresponding label as title upper to the image
      plt.title(reverse_encoding_dictionary[batch[1][n].numpy()])
      plt.axis('off')
    
  plt.savefig("sample-images.png")
  plt.close()


def plot_training(history, dir=""):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(dir+"accuracy.png")
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(dir+"loss.png")