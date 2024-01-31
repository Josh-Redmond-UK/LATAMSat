import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import glob
import shutil
import os

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