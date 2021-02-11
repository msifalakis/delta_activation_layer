# inference of UCF_TD_ResNet50

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
from video_generator.video_data_generator_UCF_mobilenet import train_generator, validation_generator  #<-- here you change the data :)
from TD_mobilenet_model import TD_MobileNet



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
weight_load_addr = os.path.join('out', "ORG_MobileNet_UCF_fine_tune", 'weights.02-0.73.hdf5')
model = TD_MobileNet(input_shape = image_shape, classes = n_classes)
model.load_weights(weight_load_addr)   
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])


### check the accuracy
a = model.evaluate(
        x = validation_generator, verbose=1,
        steps=validation_generator.__len__(),  #= validation_generator.samples // batch_size
        workers=10, use_multiprocessing=True)

np.save('inference_results/ORG_MobNet_inference',a)


#############Test results##################
# seq_size = 32 --> 