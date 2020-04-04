from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, Reshape, ELU, MaxPooling2D
from keras.optimizers import Adam


def CNNModel():
    model = Sequential()
    # model.add(Conv2D(24, (5, 5), input_shape=(240, 320, 3), strides=(2, 2), kernel_initializer='he_normal'))
    # model.add(ELU())
    # Convolution Layer 1
    # model.add(Conv2D(12, (5, 5), activation='elu', input_shape=(240, 320, 3)))
    # # Pooling Layer 1
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # # model.add(Conv2D(36, (5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    # # model.add(ELU())
    # # Convolution Layer 2
    # model.add(Conv2D(24, (5, 5), activation='relu', padding='SAME'))
    # # Pooling Layer 2
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # # model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_initializer='he_normal'))
    # # model.add(ELU())
    # # Convolution Layer 3
    # model.add(Conv2D(36, (3, 3), activation='elu'))
    # # Pooling Layer 3
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # # model.add(Dropout(0.5))
    # # model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
    # # Convolution Layer 4
    # model.add(Conv2D(48, (3, 3), activation='relu'))
    # # Pooling Layer 4
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # # (64 - 60)
    # model.add(Conv2D(60, (3, 3), strides=(1, 1), padding='valid', activation='elu'))
    #
    # model.add(Flatten())
    # # Dense layer 1
    # model.add(Dense(1164, activation='relu'))
    #
    # model.add(Dense(100, activation='elu'))
    # model.add(Dense(50, activation='relu'))
    # # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='elu'))
    # model.add(Dense(1, activation='elu'))

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(240, 320, 3)))
    model.add(MaxPooling2D(2, 2))
    # Add another convolution
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    # Now flatten the output. After this you'll just have the same DNN structure as the non convolutional version
    model.add(Flatten())
    # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='elu'))

    adam = Adam(lr=1e-2)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    return model
