from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

# Setting up a Keras model of: 4 Conv and Pool + Flat + 5 Dense
def CNNModel():
    inputShape = (240, 320, 3)
    model = Sequential()

    # Convolution Layer 1
    convLayer = Conv2D(filters=12,
                       kernel_size=(5, 5),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape)
    model.add(convLayer)
    # Pooling Layer 1
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 2
    convLayer = Conv2D(filters=24,
                       kernel_size=(5, 5),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 2
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 3
    convLayer = Conv2D(filters=36,
                       kernel_size=(3, 3),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 3
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 4
    convLayer = Conv2D(filters=48,
                       kernel_size=(3, 3),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 4
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Flatten
    model.add(Flatten())

    # Dense layer 1
    denseLayer = Dense(1164,
                       activation='elu')
    model.add(denseLayer)
    # Dense layer 2
    denseLayer = Dense(100,
                       activation='elu')
    model.add(denseLayer)
    # Dense layer 3
    denseLayer = Dense(50,
                       activation='elu')
    model.add(denseLayer)
    # Dense layer 4
    denseLayer = Dense(10,
                       activation='elu')
    model.add(denseLayer)
    # Dense layer 5
    denseLayer = Dense(1)
    model.add(denseLayer)

    # Compilation
    model.compile(Adam(lr=0.001),
                  loss='mse',
                  metrics=['accuracy', 'mse'])

    return model
