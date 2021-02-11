# -*- coding: utf-8 -*-
"""
@author: yousef21

This file implements the ResNet-50 model in its original form
"""

import tensorflow as tf
from tensorflow.python.keras import layers, Model



def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack(x, filters, blocks, stride1=2, name=None):
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x




def UCF_ORG_ResNet50(input_shape=(224,224), classes=101):

    img_input = layers.Input(shape=(input_shape[0], input_shape[1],3))

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)

    x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    #ResNet blocks
    x = stack(x, 64, 3, stride1=1, name='conv2')
    x = stack(x, 128, 4, name='conv3')
    x = stack(x, 256, 6, name='conv4')
    x = stack(x, 512, 3, name='conv5')

    #top
    x = layers.AveragePooling2D((7,7), name="avg_pool")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu", name='fc' + str(1024))(x)
    x = layers.Dropout(0.5)(x)

    #output
    predictions = layers.Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(inputs = img_input, outputs = predictions, name='ResNet50')


    return model

