from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU
from keras.optimizers import Adam

# importing cv2
import cv2

IMG_SHAPE = (240, 320, 3)


def CNNModel():
    model = Sequential()
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=IMG_SHAPE))

    model.add(Conv2D(24, (5, 5),
                     strides=(2, 2),
                     padding='valid',
                     kernel_initializer='he_normal',
                     name='conv1'))

    model.add(ELU())
    model.add(Conv2D(36, (5, 5),
                     strides=(2, 2),
                     padding='valid',
                     kernel_initializer='he_normal',
                     name='conv2'))

    model.add(ELU())
    model.add(Conv2D(48, (5, 5),
                     strides=(2, 2),
                     padding='valid',
                     kernel_initializer='he_normal',
                     name='conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     kernel_initializer='he_normal',
                     name='conv4'))

    model.add(ELU())
    model.add(Conv2D(64, (3, 3),
                     strides=(1, 1),
                     padding='valid',
                     kernel_initializer='he_normal',
                     name='conv5'))

    model.add(Flatten(name='flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
    model.add(ELU())

    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name='output', kernel_initializer='he_normal'))

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    return model
