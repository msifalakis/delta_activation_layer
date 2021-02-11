# inference of UCF_TD_ResNet50

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import datetime
from video_generator.video_data_generator_UCF_resnet import train_generator, validation_generator  #<-- here you change the data :)
from UCF_TD_ResNet50_model import UCF_TD_ResNet50



batch_size = 1
image_shape = (224, 224) #(320, 240)
n_classes = 101
seq_size = 32 #number of frames in a sequence of video
frame_stride = 1 #skip frames in a sequence
seq_stride = np.ceil(seq_size/2) #stride between current and next sequence 
opt = keras.optimizers.SGD(lr=5e-5, momentum=0.9, decay=5e-4) 

#data-set preparation

validation_generator = validation_generator (
    image_shape=image_shape, 
    batch_size=batch_size, 
    nb_frames=seq_size, 
    frame_stride=frame_stride, 
    sequence_stride=seq_stride)


#model
weight_load_addr = os.path.join('out', "ORG_ResNet50_UCF_fine_tune", 'ORG_ResNet50_UCF_fine_tune_only_head_70%_weights.h5')
model = UCF_TD_ResNet50(input_shape = image_shape, classes = n_classes)
model.load_weights(weight_load_addr)   
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])


### check the accuracy
a = model.evaluate(
        x = validation_generator, verbose=0,
        steps=1000,#validation_generator.__len__(),  #= validation_generator.samples // batch_size
        workers=10, use_multiprocessing=True)

np.save('inference_results/ORG_inference',a)


#############Test results##################
# batch_size = 4
# image_shape = (224, 224) #(320, 240)
# n_classes = 101
# frame_stride = 1 #skip frames in a sequence
# seq_stride = np.ceil(seq_size/2) #stride between current and next sequence 
# seq_size = 64 --> 4081s 1s/step - loss: 1.1695 - accuracy: 0.7362 - sp_in: 1.0000 - sp_conv1: 0.7078 - sp_conv2: 0.6325 - sp_conv3: 0.4540 - sp_conv4: 0.3931 - sp_conv5: 0.2387 - sp_fc: 0.5438
# seq_size = 1  --> accuracy: 70%
# seq_size = 32 --> loss: 1.1998 - accuracy: 0.7303 - sp_in: 1.0000 - sp_conv1: 0.7077 - sp_conv2: 0.6321 - sp_conv3: 0.4540 - sp_conv4: 0.3930 - sp_conv5: 0.2386 - sp_fc: 0.5437 - sp_all:0.5321 -operation_all: 0.5092
# seq_size = 2  -->