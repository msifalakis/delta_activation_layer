# -*- coding: utf-8 -*-
"""
@author: yousef21

This file perform the training on the UCF_ResNet50_delta_model (ResNet50 model with delta layers after every activation)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import datetime
from video_generator.video_data_generator_UCF_resnet import train_generator, validation_generator  #<-- here you change the data :)
from UCF_ResNet50_delta_model import UCF_ResNet50_delta
from UCF_TD_ResNet50_model import UCF_TD_ResNet50

#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

model_name = "ResNet50_UCF_delta"
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
sp_rate=1e-6#1e-4
thr_init=[3,3,3,4,4,4,3] #[3,1,1,1,1e-1,1e-1,1e-3]
thr_trainable = True
opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=5e-4)    

#data-set preparation
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

#with strategy.scope():
#model
# weight_load_addr = os.path.join('out', "ORG_ResNet50_UCF_fine_tune", 'ORG_ResNet50_UCF_fine_tune_only_head_70%_weights.h5')
# model_ref = UCF_TD_ResNet50(input_shape = image_shape, classes = n_classes)
# model_ref.load_weights(weight_load_addr)

#trasnfer weights to delta model and freeze (non-delta layers)
model = UCF_ResNet50_delta(input_shape = image_shape, classes = n_classes, sp_rate=sp_rate, thr_init=thr_init, thr_trainable=thr_trainable )
model.load_weights(save_dir+'/weights.08-0.60.hdf5') #<-- already trained 8 epochs
# for i in range(len(model_ref.layers)):
#     if len(model_ref.layers[i].trainable_weights)!=0: #transfer learning all layers
#         print(model.layers[i].name, model_ref.layers[i].name)
#         model.layers[i].set_weights(model_ref.layers[i].get_weights())
#         #model.layers[i].traibable=False  
#         #if model.layers[i].name[-2:]=='bn': model.layers[i].traibable=True  #unfreeze batchnorm layers



####Logging
# log_dir = os.path.join(save_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
# EarlyStopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_accuracy', min_delta=0, patience=100, verbose=1, mode='auto',
#     baseline=None, restore_best_weights=True)
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
#opt = 'adam'

model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

### check the accuracy before training thresholds
#model.evaluate(validation_generator, batch_size = batch_size, workers=10, use_multiprocessing=True)
model.fit(
        x=train_generator, epochs=num_epochs, verbose=1, callbacks=model_checkpoint, #callbacks=EarlyStopping, #callbacks=tensorboard_callback, #callbacks=csv_logger,
        validation_data=validation_generator, shuffle=True,
        initial_epoch=8,
        steps_per_epoch=train_generator.__len__(),  #= train_generator.samples // batch_size
        validation_steps=validation_generator.__len__(),
        workers=10, use_multiprocessing=True)

model.save_weights(save_dir+'/'+model_name+'.h5') # to load the weights: model.load_weights('path_to_my_model')





########Results###############
# batch_size = 4
# image_shape = (224, 224) #(320, 240)
# n_classes = 101
# seq_size = 8 #number of frames in a sequence of video
# frame_stride = 1 #skip frames in a sequence
# seq_stride = np.ceil(seq_size/2) #stride between current and next sequence 
# sp_rate=1e-4
#thr_init=[3,3,3,4,4,4,3] all layers traininble opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=5e-4)
#Epoch 2/5
#47948s 442ms/step - loss: 6.6483 - accuracy: 0.6309 - sp_in: 0.2730 - sp_conv1: 0.1068 - sp_conv2: 0.0798 - sp_conv3: 0.0521 - sp_conv4: 0.0460 - sp_conv5: 0.0512 - sp_fc: 0.0871 - 
#val_loss: 7.6733 - val_accuracy: 0.4954 - val_sp_in: 0.2772 - val_sp_conv1: 0.1092 - val_sp_conv2: 0.0823 - val_sp_conv3: 0.0536 - val_sp_conv4: 0.0468 - val_sp_conv5: 0.0536 - val_sp_fc: 0.1075
# Epoch 3/30
#108425/108425 [==============================] - 47907s 442ms/step - loss: 6.5039 - accuracy: 0.6628 - sp_in: 0.2675 - sp_conv1: 0.1079 - sp_conv2: 0.0810 - sp_conv3: 0.0517 - sp_conv4: 0.0457 - sp_conv5: 0.0498 - sp_fc: 0.0885 - 
#val_loss: 7.5590 - val_accuracy: 0.5195 - val_sp_in: 0.2724 - val_sp_conv1: 0.1109 - val_sp_conv2: 0.0830 - val_sp_conv3: 0.0538 - val_sp_conv4: 0.0471 - val_sp_conv5: 0.0528 - val_sp_fc: 0.1092
# Epoch 7/30
#110195/110195 [==============================] - 46551s 422ms/step - loss: 3.0799 - accuracy: 0.8986 - sp_in: 0.1462 - sp_conv1: 0.0609 - sp_conv2: 0.0622 - sp_conv3: 0.0443 - sp_conv4: 0.0411 - sp_conv5: 0.0383 - sp_fc: 0.0869 - 
# val_loss: 5.0642 - val_accuracy: 0.5949 - val_sp_in: 0.1534 - val_sp_conv1: 0.0656 - val_sp_conv2: 0.0649 - val_sp_conv3: 0.0465 - val_sp_conv4: 0.0426 - val_sp_conv5: 0.0397 - val_sp_fc: 0.0936

# Epoch 9/30 --> decrease sp_rate increase batch size
# batch_size = 16
# image_shape = (224, 224) #(320, 240)
# n_classes = 101
# seq_size = 2 #number of frames in a sequence of video
# frame_stride = 1 #skip frames in a sequence
# seq_stride = np.ceil(seq_size/2) #stride between current and next sequence 
# sp_rate=1e-6#1e-4
# thr_init=[3,3,3,4,4,4,3] #[3,1,1,1,1e-1,1e-1,1e-3]
# opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=5e-4)  
# Epoch 9/30   
#111080/111080 [==============================] - 44171s 398ms/step - loss: 0.2365 - accuracy: 0.9467 - sp_in: 0.1366 - sp_conv1: 0.0617 - sp_conv2: 0.0631 - sp_conv3: 0.0466 - sp_conv4: 0.0459 - sp_conv5: 0.0434 - sp_fc: 0.0908 - 
#val_loss: 2.7258 - val_accuracy: 0.5948 - val_sp_in: 0.1441 - val_sp_conv1: 0.0669 - val_sp_conv2: 0.0661 - val_sp_conv3: 0.0491 - val_sp_conv4: 0.0485 - val_sp_conv5: 0.0466 - val_sp_fc: 0.0976







































#### appendix ####

#check layers are equal
# trainable_layers = []
# for i,layer in enumerate(model.layers):
#         if len(layer.trainable_weights)!=0:
#             trainable_layers.append(i)

# trainable_layers_source_model = []
# for i,layer in enumerate(source_model.layers):
#         if len(layer.trainable_weights)!=0:
#             trainable_layers_source_model.append(i)
# for i in range(len(trainable_layers)):
#     print(i, trainable_layers[i], model.layers[trainable_layers[i]].name, source_model.layers[trainable_layers_source_model[i]].name, 
#     model.layers[trainable_layers[i]].trainable_weights[0].shape, source_model.layers[trainable_layers_source_model[i]].trainable_weights[0].shape)
#print(i, model.layers[i].name, model_ref.layers[i].name, model.layers[i].trainable_weights[0].shape, model_ref.layers[i].trainable_weights[0].shape)

### to see the layer names ####
# for i, layer in enumerate(model.layers):
#     print(i, layer.name)

# model_ORG= UCF_ORG_ResNet50(input_shape = image_shape, classes = n_classes)
# model_ORG.load_weights(load_addr)
# model_ORG.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
# model_ORG.evaluate(x=x0, y=y, batch_size = batch_size, workers=8, use_multiprocessing=True)
# p1 = np.argmax(model_TD.predict(x=x1), axis=-1)
# p0 = np.argmax(model_ORG.predict(x=x0), axis=-1)
# y  = np.argmax(y, axis=-1)
# print(np.all(p0==p1), np.sum(p0==y), np.sum(p1==y))

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# validation_generator_TD.batch_size=10000
# inp = validation_generator_TD.__getitem__(1)
# x = np.squeeze(inp[0])

#######
#a_TD = validation_generator_TD.__getitem__(1)
#a_ORG =  validation_generator._get_batches_of_transformed_samples([1])
