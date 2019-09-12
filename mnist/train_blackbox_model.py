import os

import keras
import numpy as np
import tensorflow_probability as tfp
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, BatchNormalization, Flatten
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

tfb = tfp.bijectors

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3, 4"  # specify which GPU(s) to be used
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

input_shape = x_train.shape[1:]
num_classes = 10
batch_size = 64
epochs = 200

y_train = keras.utils.to_categorical(y_train, num_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)

# Build a black-box model and get its predictions
black_box_model_weights_filename = 'black_box.h5'
black_box_model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape),
    Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'),
    MaxPool2D((2, 2)),
    Dropout(0.20),
    Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),
    Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
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

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
h = black_box_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val, y_val),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])

predicted_classes = black_box_model.predict_classes(x_test)
print("Black-box model accuracy: %.4f" % (np.count_nonzero(predicted_classes == y_test) / y_test.shape[0]))

print("Saving model...")
black_box_model.save('blackbox_model.h5')

pred_train = black_box_model.predict(x_train)
pred_test  = black_box_model.predict(x_test)
# np.save('softmax_outputs.npy', [pred_train, pred_test])