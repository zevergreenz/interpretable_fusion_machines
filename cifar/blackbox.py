# Source: https://github.com/olympus999/cifar-10-wrn

import keras.utils.np_utils as kutils
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import cifar.wide_residual_net as wrn

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings

name = 'KaggleV2.3 - WRN-28-10'

if not os.path.exists(name):
    os.makedirs(name)

lr_schedule = [60, 120, 160] # epoch_step
def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02 # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008


def random_crop(x, random_crop_size = (32,32), sync_seed=None):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]

def pad(x, pad=4):
    return np.pad(x, ((0,0), (pad,pad),(pad,pad),(0,0)), mode='reflect')

batch_size = 64
nb_epoch = 200
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

# trainX = pad(trainX)
# testX = pad(testX)

trainX = trainX.astype('float32')
trainX = (trainX - [125.3, 123.0, 113.9]) / [63.0, 62.1, 66.7]
testX = testX.astype('float32')
testX = (testX - [125.3, 123.0, 113.9]) / [63.0, 62.1, 66.7]


trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(zca_epsilon=0,
                               horizontal_flip=True,
                               fill_mode='reflect',)

generator.fit(trainX, seed=0, augment=True)

test_generator = ImageDataGenerator(zca_epsilon=0,
                               horizontal_flip=True,
                               fill_mode='reflect',)

test_generator.fit(testX, seed=0, augment=True)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
init_shape = (3, 32, 32) if 0 == 'th' else (32, 32, 3)
model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=4, k=10, dropout=0.0)

model.summary()
# plot_model(model, "WRN-28-10.png", show_shapes=True)

opt = SGD(lr=0.0008, nesterov=True, decay=0.0005)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
print("Finished compiling")
print("Allocating GPU memory")

model.load_weights('WRN-28-10 Weights.h5')
print("Model loaded.")

scores = model.evaluate_generator(test_generator.flow(testX, testY, nb_epoch), (testX.shape[0] / batch_size + 1))
print("Accuracy = %f" % (100 * scores[1]))

def train_blackbox():
    return model