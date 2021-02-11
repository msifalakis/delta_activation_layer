"""
@author: yousef21

This file perform the training on the UCF_ResNet50_L1_model (ResNet50 model with L1_regularization layers after every activation)
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
from UCF_ResNet50_L1_model import UCF_ResNet50_L1
from UCF_TD_ResNet50_model import UCF_TD_ResNet50

#strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

model_name = "ResNet50_UCF_L1"
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
sp_rate=4e-4  #<--- increasing the sparsity rate little bit to gain some sparsity 
thr_init=[3,3,3,4,4,4,3] #[3,1,1,1,1e-1,1e-1,1e-3] <-- not important here

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
weight_load_addr = os.path.join('out', "ResNet50_UCF_L1", 'weights.01-0.55.hdf5')
# model_ref = UCF_TD_ResNet50(input_shape = image_shape, classes = n_classes)
# model_ref.load_weights(weight_load_addr)

#trasnfer weights to delta model and freeze (non-delta layers)
model = UCF_ResNet50_L1(input_shape = image_shape, classes = n_classes, sp_rate=sp_rate, thr_init=thr_init)
model.load_weights(weight_load_addr)
# Transfer learning
# for i in range(len(model_ref.layers)):
#     if len(model_ref.layers[i].trainable_weights)!=0: #transfer learning all layers
#         print(model.layers[i].name, model_ref.layers[i].name)
#         model.layers[i].set_weights(model_ref.layers[i].get_weights())
        #model.layers[i].traibable=False  
        #if model.layers[i].name[-2:]=='bn': model.layers[i].traibable=True  #unfreeze batchnorm layers



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
opt = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=5e-4)    
#opt = 'adam'

model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

### check the accuracy before training thresholds
#model.evaluate(validation_generator, batch_size = batch_size, workers=10, use_multiprocessing=True)
model.fit(
        x=train_generator, epochs=num_epochs, verbose=1, callbacks=model_checkpoint, #callbacks=EarlyStopping, #callbacks=tensorboard_callback, #callbacks=csv_logger,
        validation_data=validation_generator, shuffle=True,
        initial_epoch=1,
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
# sp_rate=4e-4  #<--- increasing the sparsity rate little bit to gain some sparsity 
# Epoch 1/30
# 108425/108425 [==============================] - 35195s 325ms/step - loss: 3.0189 - accuracy: 0.8204 - sp_in: 1.0000 - sp_conv1: 0.6739 - sp_conv2: 0.5343 - sp_conv3: 0.3509 - sp_conv4: 0.2095 - sp_conv5: 0.0528 - sp_fc: 0.7509 - 
# val_loss: 26.3917 - val_accuracy: 0.5473 - val_sp_in: 1.0000 - val_sp_conv1: 0.6662 - val_sp_conv2: 0.5167 - val_sp_conv3: 0.3243 - val_sp_conv4: 0.1819 - val_sp_conv5: 0.0473 - val_sp_fc: 0.7209
# Epoch 2/30 --> overfit?!
#108425/108425 [==============================] - 35156s 324ms/step - loss: 2.1965 - accuracy: 0.9113 - sp_in: 1.0000 - sp_conv1: 0.6675 - sp_conv2: 0.5136 - sp_conv3: 0.3248 - sp_conv4: 0.1803 - sp_conv5: 0.0406 - sp_fc: 0.7605 - 
#val_loss: 28.7392 - val_accuracy: 0.4718 - val_sp_in: 1.0000 - val_sp_conv1: 0.6694 - val_sp_conv2: 0.5036 - val_sp_conv3: 0.3242 - val_sp_conv4: 0.1765 - val_sp_conv5: 0.0439 - val_sp_fc: 0.7336




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
