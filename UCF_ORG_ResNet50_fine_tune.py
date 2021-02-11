"""
@author: yousef21

This file trains the ResNet-50 model in its original form
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt 
import datetime


model_name = "ORG_ResNet50_UCF_fine_tune"
save_dir = os.path.join('out', model_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)  
    os.makedirs(os.path.join(save_dir, 'logs'))

batch_size = 32
image_shape = (224, 224) #(320, 240)
n_classes = 101


#data-set preparation
train_datagen = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
            preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
        '/mnt/space/datasets/UCF101_dataset/UCF_train_frames',
        target_size=(224, 224),
        batch_size=batch_size)          
         
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
        '/mnt/space/datasets/UCF101_dataset/UCF_test_frames',
        target_size=(224, 224),
        batch_size=batch_size)    




###### model ######
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
headModel = base_model.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
predictions = Dense(n_classes, activation='softmax')(headModel)
model = Model(inputs=base_model.input, outputs=predictions)

#Logging
log_dir = os.path.join(save_dir, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)


###
print("[INFO] training head...")
for layer in base_model.layers:
    layer.trainable = False  

num_epochs_head = 8
opt = keras.optimizers.SGD(lr=5e-4, momentum=0.9, decay=5e-4)    
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(
        x=train_generator, epochs=num_epochs_head, verbose=1, callbacks=tensorboard_callback, #callbacks=csv_logger
        validation_data=validation_generator, shuffle=True,
        initial_epoch=0, steps_per_epoch=train_generator.__len__(),  #= train_generator.samples // batch_size
        validation_steps=validation_generator.__len__(),
        workers=8, use_multiprocessing=True)

model.save_weights(save_dir+'/'+model_name+'_only_head.h5') # to load the weights: model.load_weights('path_to_my_model')

###
print("[INFO] training top conv block (conv5_block) ...")
for layer in model.layers[:143]:
   layer.trainable = False
for layer in model.layers[143:]:
   layer.trainable = True

num_epochs = 16
opt = keras.optimizers.SGD(lr=5e-5, momentum=0.9, decay=5e-4)    
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(
        x=train_generator, epochs=num_epochs, verbose=1, callbacks=tensorboard_callback, #callbacks=csv_logger,
        validation_data=validation_generator, shuffle=True,
        initial_epoch=num_epochs_head, steps_per_epoch=train_generator.__len__(),  #= train_generator.samples // batch_size
        validation_steps=validation_generator.__len__(),
        workers=8, use_multiprocessing=True)

model.save_weights(save_dir+'/'+model_name+'.h5') # to load the weights: model.load_weights('path_to_my_model')



# train_generator.next()
# validation_generator._get_batches_of_transformed_samples(range(100)) --> generate batch of 100 images

### to see the layer names ####
# for i, layer in enumerate(model.layers):
#         print(i, layer.name)

# csv_logger = keras.callbacks.CSVLogger(save_dir+'/'+model_name+'_log.csv', append=True, separator=';')
# if os.path.isfile(save_dir+'/'+model_name+'_log.csv'):
#     os.remove(save_dir+'/'+model_name+'_log.csv')


#To see the tensorboard
#tensorboard --logdir out/ResNet50_detection_split_stateless/logs/ --port 6006