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
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)  
    os.makedirs(os.path.join(save_dir, 'logs'))

batch_size = 16
image_shape = (224, 224) #(320, 240)
n_classes = 101
seq_size = 2 #number of frames in a sequence of video
frame_stride = 1 #skip frames in a sequence
seq_stride = np.ceil(seq_size/2) #stride between current and next sequence 
sp_rate=1e-3  #<--- increasing the sparsity rate little bit to gain some sparsity 
thr_init=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

# data-set preparation
train_generator = train_generator(
    image_shape=image_shape, 
    batch_size=batch_size, 
    nb_frames=seq_size, 
    frame_stride=frame_stride, 
    sequence_stride=seq_stride)

validation_generator = validation_generator (
    image_shape=image_shape, 
    batch_size=batch_size, 
    nb_frames=seq_size, 
    frame_stride=frame_stride, 
    sequence_stride=seq_stride)




model = MobileNet_delta(input_shape = image_shape, classes = n_classes, sp_rate=sp_rate, thr_init=thr_init, full_metric=False)
model.load_weights(os.path.join('out', "MobileNet_delta", 'weights.00.h5'))

# run this only first time to build the first pretrained wegihts for mobilenet_delta model 
# model_ref = MobileNet_L1(input_shape = image_shape, classes = n_classes, sp_rate=sp_rate, thr_init=thr_init, full_metric=False)
# model_ref.load_weights(os.path.join('out', "MobileNet_L1", 'weights.01-0.76.hdf5'))
# for i in range(len(model_ref.layers)):
#     if len(model_ref.layers[i].trainable_weights)!=0: #transfer learning all layers
#         print(model.layers[i].name, model_ref.layers[i].name)
#         model.layers[i].set_weights(model_ref.layers[i].get_weights())
# model.save_weights(save_dir+'/'+'weights.00'+'.h5') # to load the weights: model.load_weights('path_to_my_model')



####Logging
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    save_dir+'/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5',
    monitor="val_accuracy",
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
    mode="max",
    save_freq="epoch",
)

num_epochs = 30
opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=5e-4)    

model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

### check the accuracy before training thresholds
#model.evaluate(validation_generator, batch_size = batch_size, workers=10, use_multiprocessing=True)

model.fit(
        x=train_generator, epochs=num_epochs, verbose=1, callbacks=model_checkpoint, #callbacks=EarlyStopping, #callbacks=tensorboard_callback, #callbacks=csv_logger,
        validation_data=validation_generator, shuffle=True,
        initial_epoch=0,
        steps_per_epoch=train_generator.__len__(),  #= train_generator.samples // batch_size
        validation_steps=validation_generator.__len__(),
        workers=10, use_multiprocessing=True)

model.save_weights(save_dir+'/'+model_name+'.h5') # to load the weights: model.load_weights('path_to_my_model')





########Results###############


