import tensorflow as tf
from src.utils import *
from src.model import *

image_shape = (64,64,3)
num_classes = 19
model = make_model(input_shape=image_shape, num_classes=num_classes)

model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/rgb_model",
    save_weights_only=False,
    monitor="accuracy",
    mode="max",
    save_best_only=True,
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

train_dataset = get_rgb_dataset('latamSatData/datasetRGB_relabel/', shuffle_size='full')
#train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality()).batch(32)

test = train_dataset.as_numpy_iterator().next()[0]


model.fit(train_dataset, epochs=10, verbose=1, callbacks=[model_callback])