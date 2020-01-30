# SceneEDNet
# Fully convolutional neural network for scene flow estimation


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, \
     Cropping2D, Activation, Conv2DTranspose, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras import losses
from keras import optimizers
from keras.models import load_model

from keras.utils import multi_gpu_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import History

from sceneedclass import DataGenerator
from readfiles import DataRead
from sceneflow import SceneFlow

train_path = "training set path"
val_path = "validation set path"
Nepoch = 100

# total number of train and validation folders
train_steps = 'number of folders in training set'
val_steps  = 'number of folders in validation set'

# learning rate to keep
learning_rate = 'learinng rate'
#lrdecay = 0.0
lrdecay       = learning_rate/Nepoch

train_folder_list = range(0, train_steps)
val_folder_list   = range(0, val_steps)


# Data Generation for training and validation

training_generator   = DataGenerator().generate(train_path, train_folder_list)
validation_generator = DataGenerator().generate(val_path, val_folder_list)
#test_generator     = DataGenerator().generate(test_folder_list)
print learning_rate


def scene_model():

    inimage = Input(shape=(540, 960, 12))
    conv0 = Conv2D(64,   (3, 3), name = 'conv0',   strides = 2, padding='same')(inimage)
    conv0 = LeakyReLU()(conv0)
    conv1 = Conv2D(128,  (3, 3), name = 'conv1', strides = 2, padding='same')(conv0)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv2D(256,  (3, 3), name = 'conv2', strides = 2, padding='same')(conv1)
    conv2 = LeakyReLU()(conv2)
    conv3 = Conv2D(512,  (3, 3), name = 'conv3', strides = 2, padding='same')(conv2)
    conv3 = LeakyReLU()(conv3)
    conv4 = Conv2D(1024, (3, 3), name = 'conv4', strides = 1, padding='same')(conv3)
    conv4 = LeakyReLU()(conv4)

    conv5 = Conv2D(1024, (3, 3), name = 'conv5', strides = 1, padding='same')(conv4)
    conv5 = LeakyReLU()(conv5)
    up1   = UpSampling2D((2,2))(conv5)
    conv6 = Conv2D(512, (3, 3), name = 'conv6', strides = 1, padding='same')(up1)
    conv6 = LeakyReLU()(conv6)
    up2   = UpSampling2D((2,2))(conv6)
    conv7 = Conv2D(256, (3, 3), name = 'conv7', strides = 1, padding='same')(up2)
    conv7 = LeakyReLU()(conv7)
    up3   = UpSampling2D((2,2))(conv7)
    conv8 = Conv2D(128, (3, 3), name = 'conv8', strides = 1, padding='same')(up3)
    conv8 = LeakyReLU()(conv8)
    up4   = UpSampling2D((2,2))(conv8)
    out   = Cropping2D(cropping=((4,0),(0,0)))(up4)
    conv9 = Conv2D(64, (3, 3), name = 'conv9', strides = 1, padding='same')(out)
    conv9 = LeakyReLU()(conv9)
    output = Conv2D(3, (3, 3), name = 'output', strides = 1, padding='same')(conv9)
    #outimage = LeakyReLU()(outimage)
    model = Model(inputs=inimage, outputs=output)

    model.summary()

    return model


with tf.device('/cpu:0'):
    model = scene_model()

parallel_model = multi_gpu_model(model, gpus=2)

# Loss function and optimizer

adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
       epsilon=None, decay=lrdecay, amsgrad=False)

