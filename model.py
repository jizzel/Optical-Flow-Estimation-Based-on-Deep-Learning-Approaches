from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU
from keras.optimizers import Adam


def CNNModel():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), input_shape=(240, 320, 3), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(Conv2D(36, (5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal'))
    model.add(Flatten())
    model.add(ELU())
    model.add(Dense(100, kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer='he_normal'))
    # model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(ELU())
    model.add(Dense(1, kernel_initializer='he_normal'))

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy', 'mse'])

    return model
