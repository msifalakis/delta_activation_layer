# -*- coding: utf-8 -*-
"""
@author: yousef21

This file perform the training on the UCF_ResNet50_delta_model (ResNet50 model with delta layers after every activation)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import datetime
from video_generator.video_data_generator_UCF_resnet import train_generator, validation_generator  #<-- here you change the data :)
from UCF_ResNet50_L1_model import UCF_ResNet50_L1
from UCF_TD_ResNet50_model import UCF_TD_ResNet50



model_name = "ResNet50_UCF_L1"
save_dir = os.path.join('out', model_name)


batch_size = 1
seq_size = 1 #number of frames in a sequence of video
image_shape = (224, 224) #(320, 240)
n_classes = 101
frame_stride = 1 #skip frames in a sequence
seq_stride = np.ceil(seq_size/2) #stride between current and next sequence 

#not important here
sp_rate=0
thr_init=[0,0,0,0,0,0,0]
opt = keras.optimizers.SGD(lr=0, momentum=0, decay=0) 

#data-set preparation
validation_generator = validation_generator (
    image_shape=image_shape, 
    batch_size=batch_size, 
    nb_frames=seq_size, 
    frame_stride=frame_stride, 
    sequence_stride=seq_stride)



model = UCF_ResNet50_L1(input_shape = image_shape, classes = n_classes, sp_rate=sp_rate, thr_init=thr_init)
model.load_weights(save_dir+'/weights.17-0.63.hdf5')
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])


### check the accuracy
a = model.evaluate(
        x = validation_generator, verbose=0,
        steps=1000,#validation_generator.__len__(),  #= validation_generator.samples // batch_size
        workers=10, use_multiprocessing=True)

np.save('inference_results/L1_inference_signel_frame',a)



# Some results:
# model.load_weights(save_dir+'/weights.17-0.63.hdf5.hdf5')
# seq_size = 32 --> loss: 1.3734 - accuracy: 0.6538 - sp_in: 1.0000 - sp_conv1: 0.6523 - sp_conv2: 0.4927 - sp_conv3: 0.3054 - sp_conv4: 0.1726 - sp_conv5: 0.0490 - sp_fc: 0.7099
# seq_size = 2  -->
