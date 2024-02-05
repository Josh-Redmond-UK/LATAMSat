import tensorflow as tf
from src.utils import *
from src.model import *
import datetime
import pickle


#image_shape=(64,64,3)
num_classes=19
#model_path="models/rgb/"


image_shape = (64,64,12)
num_classes = 19
model = make_model(input_shape=image_shape, num_classes=num_classes)

model_path = f"models/ms_model/{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_weights_only=False,
    monitor="accuracy",
    mode="max",
    save_best_only=True,
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

train_dataset = get_dataset('latamSatData/datasetRGB_rescaled/*/*/*.tif', mode="ms", shuffle_size='full')


history = model.fit(train_dataset, epochs=10, verbose=1, callbacks=[model_callback])

pickle.dump(history.history, open(f'{model_path}/history.pkl', 'wb'))
