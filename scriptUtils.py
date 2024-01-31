import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import glob
import shutil
import os
import pandas as pd
import rasterio as rio
import numpy as np

classDictionary = {30: 'Herbaceous Vegetation',
20: 'Shrubs',
40: 'Agricultural Land',
50: 'Urban Areas',
60: 'Bare Earth/Sparse Vegetation',
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

label_codes = classDictionary.keys()
label_int_rep = [i for i in range(len(label_codes))]
label_int_dict = dict(zip(label_codes, label_int_rep))


classes = glob.glob('latamSatData/datasetRGB_relabel/*/*')
classes = [c.split('/')[-1] for c in classes]
idxKeys = [i for i in range(len(classes))]

classnameDict = dict(zip(idxKeys, classes))


def onehot_encode(x, y):
    one_hot = tf.one_hot(y, 19)
    return x, one_hot 

def train_test_split_(dataset, test_size):
    pct_size = dataset.cardinality().numpy() / 100 
    test_size = round(pct_size * test_size)
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)
    return train_dataset, test_dataset


def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 768]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)

def train_model(model, train_dataset, test_dataset, num_epochs):
    trainHistory = model.fit(train_dataset, validation_data=test_dataset, epochs=num_epochs)
    return trainHistory

def relabel_filepath(file_path, new_label):
    file_path = file_path.split('/')
    file_path[-2]=new_label
    file_path = '/'.join(file_path)
    return file_path

def move_file(old_path, new_path):
    shutil.move(old_path, new_path)


def train_on_data(model, train, test, callbacks = None, epochs=1):
    train_history = model.fit(train, validation_data = test, epochs=1, callbacks=callbacks)
    history = train_history.history
    history_frame = pd.DataFrame([history['loss'], history['val_loss'], history['accuracy'], history['val_accuracy']],
                    index = ['Loss', 'Val_Loss', 'Accuracy', 'Val_Accuracy'])

    return model, history_frame


def get_probs_and_featurs(model, images):
    feature_extraction = tf.keras.Model(model.input, model.layers[-3].output)

    classProb = np.squeeze(np.array(model.predict(images, verbose=0)))
    predFeat= np.squeeze(np.array(feature_extraction.predict(images, verbose=0)))

    return classProb, predFeat



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
