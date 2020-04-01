from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, BatchNormalization, ELU, MaxPooling2D
from keras.optimizers import Adam


def CNNModel():
    model = Sequential()
    # model.add(Conv2D(24, (5, 5), input_shape=(240, 320, 3), strides=(2, 2), kernel_initializer='he_normal'))
    # model.add(ELU())
    # Convolution Layer 1
    model.add(Conv2D(12, (5, 5), strides=(1, 1), activation='elu', input_shape=(240, 320, 3), kernel_initializer='he_normal'))
    # Pooling Layer 1
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(36, (5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    # model.add(ELU())
    # Convolution Layer 2
    model.add(Conv2D(24, (5, 5), activation='elu', padding='SAME',  kernel_initializer='he_normal'))
    # Pooling Layer 2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    # model.add(ELU())
    # Convolution Layer 3
    model.add(Conv2D(36, (3, 3), activation='elu', kernel_initializer='he_normal'))
    # Pooling Layer 3
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Dropout(0.5))
    # model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
    # Convolution Layer 4
    model.add(Conv2D(48, (3, 3), activation='elu', kernel_initializer='he_normal'))
    # Pooling Layer 4
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='elu', kernel_initializer='he_normal'))

    model.add(Flatten())
    model.add(ELU())
    # Dense layer 1
    model.add(Dense(1164, activation='elu'))

    model.add(Dense(100, kernel_initializer='he_normal', activation='elu'))
    model.add(Dense(50, kernel_initializer='he_normal', activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_initializer='he_normal', activation='elu'))
    model.add(Dense(1, kernel_initializer='he_normal', activation='elu'))

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    return model
