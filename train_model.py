import tensorflow as tf
from src.utils import *
from src.model import *
import datetime
import pickle


#image_shape=(64,64,3)

mode = "rgb"

if mode == "rgb":
    image_shape=(64,64,3)
    model_path = f"models/rgb_model/{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    dataset_glob_pattern = 'latamSatData/datasetRGB_relabel/**/**/*.png'

else:
    image_shape = (64,64,12)
    model_path = f"models/ms_model/{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    dataset_glob_pattern = 'latamSatData/DownloadedDataset/**/**/*.tif'



num_classes = 19
model = get_efficient_net(input_shape=image_shape, num_classes=num_classes)


model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_weights_only=False,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", "sparse_categorical_accuracy"]
)

train, test = get_dataset(dataset_glob_pattern, mode=mode, shuffle=True)

train_batch_ = next(iter(train))
#print(train_batch_)
show_batch(train_batch_)

history = model.fit(train, validation_data=test, epochs=50, verbose=1, callbacks=[model_callback])

plot_training(history, model_path+'/')

pickle.dump(history.history, open(f'{model_path}/history.pkl', 'wb'))
