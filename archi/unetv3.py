
import os
#from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import math
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
#from model import *
import numpy as np
from matplotlib import pyplot as plt
import PIL
from PIL import ImageOps
from PIL import Image
import re
from keras import backend as K
from keras import utils
from tensorflow.python.ops import array_ops
import cv2


import tensorflow as tf
import datetime
from keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp

import numpy as np
import pandas as pd


from tensorboard.plugins.hparams import api as hp

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#GPU check
physical_devices = tf.config.experimental.list_physical_devices('GPU')

for i in range(len(physical_devices)):
    config = tf.config.experimental.set_memory_growth(physical_devices[i], True)

batch_size = 10  # for example, we generate data for 10 samples
img_size = (128, 128)  # depth x height x width
num_channels = 9  # as mentioned

# Create random data using numpy
dummy_data = np.random.randn(batch_size, *img_size, num_channels).astype(np.float32)



# model_types = ['unet', 'xception', 'deeplabv3plus']
model_dict = {
    'type' : 'unet',
    'params_str' : 'base_2',
    'seed' : 1,
    'epochs' : 5,
    'test_only' : False,
    'num_channels':9
}



#pre set up
num_channels = model_dict['num_channels']


def unet_model(img_size, num_classes=2, pretrained_weights = None):
    #inputs = Input(input_size)
    inputs = keras.Input(shape=img_size + (num_channels,))

    input_channels = layers.Lambda(lambda x: [x[:, :, :, i:i+3] for i in range(0, num_channels, 3)])(inputs)

    conv_results = []
    #storing the layer for each path
    conv1_layers = []
    conv2_layers = []
    conv3_layers = []
    conv4_layers = []


    for i in range(3):
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_channels[i])
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv1_layers.append(conv1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv2_layers.append(conv2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv3_layers.append(conv3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = layers.Dropout(0.5)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        conv4_layers.append(drop4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = layers.Dropout(0.5)(conv5)


        conv_results.append(drop5)
    # Concatenate the convolution results
    merged_conv = layers.Concatenate(axis=-1)(conv_results)

    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(merged_conv))
    merge6 = layers.Add()([conv4_layers[-2],up6])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.Add()([conv3_layers[-2],up7])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.Add()([conv2_layers[-2],up8])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.Add()([conv1_layers[-2],up9])
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = layers.Conv2D(2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = layers.Conv2D(num_classes, 1, activation = 'softmax')(conv9)
    conv10 = layers.Conv2D(1, 1, activation = 'sigmoid', padding='same')(conv9)


    model = keras.Model(inputs, conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
