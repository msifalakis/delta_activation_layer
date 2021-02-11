"""
@author: yousef21

This file perform the training on the UCF_ResNet50_L1_model (ResNet50 model with L1_regularization layers after every activation)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import datetime
from video_generator.video_data_generator_UCF_mobilenet import train_generator, validation_generator  #<-- here you change the data :)
from mobilenet_delta_model import MobileNet_delta
from mobilenet_L1_model import MobileNet_L1


model_name = "MobileNet_delta"
save_dir = os.path.join('out', model_name)

batch_size = 1
image_shape = (224, 224) #(320, 240)
n_classes = 101
seq_size = 32 #number of frames in a sequence of video
frame_stride = 1 #skip frames in a sequence
seq_stride = np.ceil(seq_size/2) #stride between current and next sequence 
sp_rate=4e-4  #<--- increasing the sparsity rate little bit to gain some sparsity 
thr_init=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]



model = MobileNet_delta(input_shape = image_shape, classes = n_classes, sp_rate=sp_rate, thr_init=thr_init, full_metric=False)
model.load_weights(os.path.join('out', "MobileNet_delta", 'weights.07-0.70.hdf5'))


# data-set preparation
validation_generator = validation_generator (
    image_shape=image_shape, 
    batch_size=batch_size, 
    nb_frames=seq_size, 
    frame_stride=frame_stride, 
    sequence_stride=seq_stride)






opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=5e-4)    
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

### check the accuracy before training thresholds
a = model.evaluate(
        x = validation_generator, verbose=1,
        steps=validation_generator.__len__(),  #= validation_generator.samples // batch_size
        workers=10, use_multiprocessing=True)

np.save('inference_results/MobNet_delta__inference',a)




########Results###############

