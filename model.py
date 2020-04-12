from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, ELU, MaxPooling2D
from keras.optimizers import Adam

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, ELU, TimeDistributed, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import MaxPooling2D

# def CNNModel():
#     model = Sequential()
#     # Convolution Layer 1
#     model.add(Conv2D(12, (3, 3), activation='elu', input_shape=(240, 320, 3)))
#     # Pooling Layer 1
#     model.add(MaxPooling2D(2, 2))
#
#     # Convolution Layer 2
#     model.add(Conv2D(24, (3, 3), activation='elu'))
#     # Pooling Layer 2
#     model.add(MaxPooling2D(2, 2))
#
#     # model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_initializer='he_normal'))
#     # model.add(ELU())
#     # Convolution Layer 3
#     model.add(Conv2D(36, (3, 3), activation='elu'))
#     # Pooling Layer 3
#     model.add(MaxPooling2D(2, 2))
#
#     # model.add(Dropout(0.5))
#     # model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
#     # Convolution Layer 4
#     model.add(Conv2D(48, (3, 3), activation='elu'))
#     # Pooling Layer 4
#     model.add(MaxPooling2D(2, 2))
#
#     # (64 - 60)
#     model.add(Conv2D(60, (3, 3), activation='elu'))
#     model.add(MaxPooling2D(2, 2))
#
#     model.add(Flatten())
#     # Dense layer 1
#     model.add(Dense(1164, activation='elu'))
#
#     model.add(Dense(100, activation='elu'))
#     model.add(Dense(50, activation='elu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(10, activation='elu'))
#     model.add(Dense(1, activation='elu'))
#
#     adam = Adam(lr=1e-4)
#     model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
#
#     return model


# def nvidia_model():
#     model = Sequential()
#     # normalization
#     # perform custom normalization before lambda layer in network
#     model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(240, 320, 3)))
#
#     model.add(Conv2D(24, (5, 5),
#                      strides=(2, 2),
#                      padding='valid',
#                      kernel_initializer='he_normal',
#                      name='conv1'))
#
#     model.add(ELU())
#     model.add(Conv2D(36, (5, 5),
#                      strides=(2, 2),
#                      padding='valid',
#                      kernel_initializer='he_normal',
#                      name='conv2'))
#
#     model.add(ELU())
#     model.add(Conv2D(48, (5, 5),
#                      strides=(2, 2),
#                      padding='valid',
#                      kernel_initializer='he_normal',
#                      name='conv3'))
#     model.add(ELU())
#     model.add(Dropout(0.5))
#     model.add(Conv2D(64, (3, 3),
#                      strides=(1, 1),
#                      padding='valid',
#                      kernel_initializer='he_normal',
#                      name='conv4'))
#
#     model.add(ELU())
#     model.add(Conv2D(64, (3, 3),
#                      strides=(1, 1),
#                      padding='valid',
#                      kernel_initializer='he_normal',
#                      name='conv5'))
#
#     model.add(Flatten(name='flatten'))
#     model.add(ELU())
#     model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
#     model.add(ELU())
#     model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
#     model.add(ELU())
#     model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
#     model.add(ELU())
#
#     # do not put activation at the end because we want to exact output, not a class identifier
#     model.add(Dense(1, name='output', kernel_initializer='he_normal'))
#
#     adam = Adam(lr=1e-4)
#     model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
#
#     return model
IMG_SHAPE = (240, 320, 3)


def CNNModel():
    inputs = Input(shape=IMG_SHAPE)
    inputs1 = Lambda(lambda x: x / 127.5 - 1, input_shape=IMG_SHAPE)(inputs)

    conv1 = Conv2D(24, (5, 5), padding="valid")(inputs1)
    act1 = Activation(ELU())(conv1)
    conv2 = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(act1)
    act2 = Activation(ELU())(conv2)
    drop1 = Dropout(0.5)(act2)
    conv3 = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(drop1)
    act3 = Activation(ELU())(conv3)
    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid")(act3)
    act4 = Activation(ELU())(conv4)
    conv5 = Conv2D(64, (3, 3), padding="valid")(act4)

    flat1 = Flatten()(conv5)
    act4 = Activation(ELU())(flat1)
    dense1 = Dense(100)(act4)
    act5 = Activation(ELU())(dense1)
    dense2 = Dense(50)(act5)
    act6 = Activation(ELU())(dense2)
    dense4 = Dense(10)(act6)
    act8 = Activation(ELU())(dense4)
    output = Dense(1)(act8)

    model = Model(inputs, output)
    adam = Nadam()
    model.compile(optimizer=adam, loss='mse')

    print(model.summary())
    return model
