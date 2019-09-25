import os

import keras
import numpy as np
import tensorflow_probability as tfp
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, BatchNormalization, Flatten
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

tfb = tfp.bijectors

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3, 4"  # specify which GPU(s) to be used
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings


# MNIST dataset
f = h5py.File('synthetic_dataset.hdf5', 'r')
x_train = np.array(f["train/x"])
y_train = np.array(f["train/y"])
x_test = np.array(f["test/x"])
y_test = np.array(f["test/y"])
true_alpha = np.array(f["alpha"])
f.close()
y_train = keras.utils.to_categorical(y_train)

input_shape = x_train.shape[1:]
num_classes = y_train.shape[1]
batch_size = 64
epochs = 1000


# Build a black-box model and get its predictions
black_box_model_weights_filename = 'black_box.h5'
black_box_model = keras.Sequential([
    Dense(3, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

black_box_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)

h = black_box_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

predicted_classes = black_box_model.predict_classes(x_test)
print("Black-box model accuracy: %.4f" % (np.count_nonzero(predicted_classes == y_test) / y_test.shape[0]))

print("Saving model...")
black_box_model.save('blackbox_model.h5')

pred_train = black_box_model.predict(x_train)
pred_test  = black_box_model.predict(x_test)
np.save('softmax_outputs.npy', [pred_train, pred_test])