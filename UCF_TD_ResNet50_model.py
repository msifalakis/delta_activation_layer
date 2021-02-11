
"""
@author: yousef21

This file implements the UCF_ResNet50_delta_model (ResNet50 model with delta layers after every activation)
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from custom_layers import L1_regulizer



def block1(x, filters, next_filter, kernel_size=3, stride=1, conv_shortcut=True, name=None, sp_rate=1e-5, thr_init=1e-1): # next_filter to calculate n_outputs 

    n_spikes = 0
    n_neurons=0

    if conv_shortcut:
        shortcut = layers.TimeDistributed(layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv'), name=name + '_0_conv')(x)
        shortcut = layers.TimeDistributed(layers.BatchNormalization(epsilon=1.001e-5, name=name + '_0_bn'), name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.TimeDistributed(layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv'), name=name + '_1_conv')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn'), name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)
    
    x, s, n = L1_regulizer(sp_rate=sp_rate, thr_init=thr_init, name=name + '_1_delta', n_outputs=filters*kernel_size*kernel_size, show_metric=True)(x) #<--
    n_spikes += s
    n_neurons +=n

    x = layers.TimeDistributed(layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv'), name=name + '_2_conv')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn'), name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    
    x, s, n = L1_regulizer(sp_rate=sp_rate, thr_init=thr_init, name=name + '_2_delta', n_outputs=4*filters*1*1, show_metric=True)(x) #<--
    n_spikes += s
    n_neurons +=n

    x = layers.TimeDistributed(layers.Conv2D(4 * filters, 1, name=name + '_3_conv'), name=name + '_3_conv')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(epsilon=1.001e-5, name=name + '_3_bn'), name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    
    x, s, n = L1_regulizer(sp_rate=sp_rate, thr_init=thr_init, name=name + '_out_delta', n_outputs=next_filter, show_metric=True)(x) #<--
    n_spikes += s
    n_neurons +=n

    return x,  n_spikes, n_neurons


def stack(x, filters, blocks, next_filter, stride1=2, name=None, sp_rate=1e-5, thr_init=1e-1):
    n_spikes = 0
    n_neurons=0

    x,s,n = block1(x, filters, next_filter=filters, stride=stride1, name=name + '_block1', sp_rate=sp_rate, thr_init=thr_init)
    n_spikes += s
    n_neurons +=n

    for i in range(2, blocks + 1):
        n_filter=filters
        if(i==blocks): n_filter=next_filter
        x,s,n = block1(x, filters, next_filter=n_filter, conv_shortcut=False, name=name + '_block' + str(i), sp_rate=sp_rate, thr_init=thr_init)
        n_spikes += s
        n_neurons +=n
    return x, n_spikes, n_neurons




def UCF_TD_ResNet50(input_shape=(224,224), classes=101, sp_rate=1e-5, thr_init=[0,0,0,0,0,0,0]):

    img_input = layers.Input(shape=(None, input_shape[0], input_shape[1],3))
    x, s_in, n_in = L1_regulizer(sp_rate=0, thr_init=thr_init[0], name='inp__delta', n_outputs=64*7*7, show_metric=True)(img_input) #<-- just for metrics
   
    x = layers.TimeDistributed(layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad'), name='conv1_pad')(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv'), name='conv1_conv')(x)

    x = layers.TimeDistributed(layers.BatchNormalization(epsilon=1.001e-5, name='conv1_bn'), name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x, s_conv1, n_conv1 = L1_regulizer(sp_rate=sp_rate, thr_init=thr_init[1], name='conv1_delta', n_outputs=64*5*1*1, show_metric=True)(x) #<--


    x = layers.TimeDistributed(layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad'), name='pool1_pad')(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(3, strides=2, name='pool1_pool'), name='pool1_pool')(x)

    #ResNet blocks
    x, s_conv2, n_conv2 = stack(x,  64, 3, next_filter=128*5, stride1=1, name='conv2', sp_rate=sp_rate, thr_init=thr_init[2])
    x, s_conv3, n_conv3 = stack(x, 128, 4, next_filter=256*5, name='conv3', sp_rate=sp_rate, thr_init=thr_init[3])
    x, s_conv4, n_conv4 = stack(x, 256, 6, next_filter=512*5, name='conv4', sp_rate=sp_rate, thr_init=thr_init[4])
    x, s_conv5, n_conv5 = stack(x, 512, 3, next_filter=1024, name='conv5', sp_rate=sp_rate, thr_init=thr_init[5])

    #top
    x = layers.TimeDistributed(layers.AveragePooling2D((7,7), name="avg_pool"), name="avg_pool")(x)
    x = layers.TimeDistributed(layers.Flatten(), name='flatten')(x)
    x = layers.Dense(1024, activation="relu", name='fc' + str(1024))(x)
    x, s_fc, n_fc = L1_regulizer(sp_rate=sp_rate, thr_init=thr_init[6], name='fc'+str(1024)+'_delta', n_outputs=classes, show_metric=True)(x) #<--

    x = layers.Dropout(0.5)(x)

    #output
    predictions = layers.Dense(classes, activation='softmax', name='fc' + str(classes))(x)

    #predictions = layers.Activation('softmax', name='predictions')(x)

    avg = tf.math.reduce_mean(predictions, axis=1) #average predictions over time
    # Create model.
    model = Model(inputs = img_input, outputs = avg, name='ResNet50')

    s_all = s_in + s_conv1 + s_conv2 + s_conv3 + s_conv4 + s_conv5 + s_fc
    n_all = n_in + n_conv1 + n_conv2 + n_conv3 + n_conv4 + n_conv5 + n_fc

    # model.add_metric(s_in/n_in,       name='sp_in',    aggregation='mean') #moving average number of spikes
    # model.add_metric(s_conv1/n_conv1, name='sp_conv1', aggregation='mean') #moving average number of spikes
    # model.add_metric(s_conv2/n_conv2, name='sp_conv2', aggregation='mean') #moving average number of spikes
    # model.add_metric(s_conv3/n_conv3, name='sp_conv3', aggregation='mean') #moving average number of spikes
    # model.add_metric(s_conv4/n_conv4, name='sp_conv4', aggregation='mean') #moving average number of spikes
    # model.add_metric(s_conv5/n_conv5, name='sp_conv5', aggregation='mean') #moving average number of spikes    
    # model.add_metric(s_fc/n_fc,       name='sp_fc',    aggregation='mean') #moving average number of spikes
    # model.add_metric(s_all/n_all,     name='sp_all',    aggregation='mean') #moving average number of spikes

    return model


