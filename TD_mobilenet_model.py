# modified mobilenet network from keras to be time Distributed


"""MobileNet v1 models for Keras.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).

The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------

Reference paper:
  - [MobileNets: Efficient Convolutional Neural Networks for
     Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend, layers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers, TimeDistributed
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export



def TD_MobileNet(input_shape=(224,224,3), classes=101):
    alpha = 1.0
    depth_multiplier = 1

    img_input = layers.Input(shape=(None, input_shape[0], input_shape[1],3))

    x = td_conv_block(img_input, 32, alpha, strides=(2, 2))
    x = td_depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = td_depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = td_depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = td_depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = td_depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = td_depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = td_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = td_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = td_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = td_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = td_depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = td_depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = td_depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    x = TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = TimeDistributed(layers.Reshape((1, 1, 1024), name='reshape_1'), name='reshape_1')(x)
    x = layers.Dropout(0.001, name='dropout')(x)
    x = TimeDistributed(layers.Conv2D(classes, (1, 1), padding='same', name='conv_preds'), name='conv_preds')(x)
    x = TimeDistributed(layers.Reshape((classes,), name='reshape_2'), name='reshape_2')(x)
    predictions = layers.Activation(activation='softmax', name='predictions')(x)

    avg = tf.math.reduce_mean(predictions, axis=1) #average predictions over time

    # Create model.
    model = training.Model(img_input, avg, name='mobilenet')

    return model


def td_conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = TimeDistributed(layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad'),name='conv1_pad')(inputs)
    x = TimeDistributed(layers.Conv2D(filters,kernel,padding='valid',use_bias=False,strides=strides,name='conv1'),name='conv1')(x)
    x = TimeDistributed(layers.BatchNormalization(axis=channel_axis, name='conv1_bn'), name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)
    return x


def td_depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1): x = inputs
    else: x = TimeDistributed(layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id), name='conv_pad_%d' % block_id)(inputs)
    
    x = TimeDistributed(layers.DepthwiseConv2D((3, 3), padding='same' if strides == (1, 1) else 'valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name='conv_dw_%d' % block_id), name='conv_dw_%d' % block_id)(x)
    x = TimeDistributed(layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id), name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = TimeDistributed(layers.Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id), name='conv_pw_%d' % block_id)(x)
    x = TimeDistributed(layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id), name='conv_pw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
    return x


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf') # Will scale pixels between -1 and 1, sample-wise.




