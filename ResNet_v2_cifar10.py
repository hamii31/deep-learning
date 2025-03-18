from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import math


# Data preprocessing
# load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions.
input_shape = x_train.shape[1:]

# normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# if subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

num_classes = 10

# convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# training parameters
batch_size = 32 # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
filters = 16
kernel_size = 3
kernel_initializer = 'he_normal'
strides = 1
padding='same',
hidden_layer_activation = 'relu'

# Defining the Input ResNet v2 Layer (32x32,16)

# ResNet v2 uses Conv2D-Bn-RelU for the input:
input_layer = Input(shape=input_shape)
conv_layer_1 = Conv2D(filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              padding=padding,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=l2(1e-4))
bn_layer_1 = BatchNormalization()(input_layer)
relu_layer_1 = Activation(hidden_layer_activation)(input_layer)

# Defining the Residual Blocks
num_res_blocks = 2

# ResNet v2 uses BN-ReLU-Conv2D Blocks - Stacks of (1 x 1)-(3 x 3)-(1 x 1) (Bottleneck Layers) 
# Stage 1
# First Residual Block
stage_1_updated_filters = filters * 4 # increasing the units each stage (deeper layers each stage)
# Initialize First BRU (Bottleneck Residual Unit)
# Residual Layer 1 (No Bn and ReLU)
bru_1_res_layer_1 = Conv2D(filters=filters,
              kernel_size=1,
              strides=strides,
              padding=padding,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=l2(1e-4))(relu_layer_1)

# Residual Layer 2
bru_1_res_layer_2 = BatchNormalization()(bru_1_res_layer_1)
bru_1_res_layer_2 = Activation(hidden_layer_activation)(bru_1_res_layer_2)
bru_1_res_layer_2 = Conv2D(filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              padding=padding,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=l2(1e-4))(bru_1_res_layer_2)

# Residual Layer 3
bru_1_res_layer_3 = BatchNormalization()(bru_1_res_layer_2)
bru_1_res_layer_3 = Activation(hidden_layer_activation)(bru_1_res_layer_3)
bru_1_res_layer_3 = Conv2D(filters=stage_1_updated_filters,
              kernel_size=1,
              strides=strides,
              padding=padding,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=l2(1e-4))(bru_1_res_layer_3)

# Linear projection residual shortcut connection to match dimensions
lp_res_short_connection = Conv2D(filters=stage_1_updated_filters,
                                kernel_size=1,
                                strides=strides,
                                padding=padding,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=l2(1e-4))(relu_layer_1)

# Adding the shortcut connection to the output of the last layer
res_block_1_output = add([lp_res_short_connection, bru_1_res_layer_3])
filters = stage_1_updated_filters

# Initialize Second BRU
