"""
This is the video data generator implementation
Make sure to adjust the addresses of the dataset here
For the whole UCF 101 dataset
"""
#%%
import csv
import numpy as np
import cv2
import os.path
import random
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from video_generator.sliding_amir import SlidingFrameGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

def get_classes(class_limit=None):
    classes = []
    with open(os.path.join('dataset_preparation/', 'UCF_classInd.txt'), 'r') as fin:
        reader = [read.strip() for read in fin.readlines()]
        for p in reader:
            line = p.split(' ')
            classes.append(line[1])

    # Sort them.
    classes = sorted(classes)

    # Return.
    if class_limit is not None:
        return classes[:class_limit]
    else:
        return classes


def train_generator(image_shape=(224, 224), batch_size=32, class_limit=None, nb_frames=16, frame_stride=1, sequence_stride=1):
    train_datagen = ImageDataGenerator(
                rotation_range=30,
	            zoom_range=0.15,
	            width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest")

    classes = get_classes(class_limit)
    glob_pattern_train = os.path.join('dataset_preparation/', 'UCF_train','{classname}','*.avi')

    generator = SlidingFrameGenerator(
        sequence_stride = sequence_stride,
        frame_stride= frame_stride,
        rescale=1.0,
        classes=classes, 
        glob_pattern=glob_pattern_train,
        nb_frames=nb_frames,
        shuffle=True,
        batch_size=batch_size,
        target_shape=image_shape,
        nb_channel=3,
        transformation=train_datagen,
        use_frame_cache=False,
        preprocessing_type='MobileNet')

    return generator

def validation_generator(image_shape=(224, 224), batch_size=32, class_limit=None, nb_frames=16, frame_stride=1, sequence_stride=1):

    test_datagen = ImageDataGenerator()

    classes = get_classes(class_limit)
    glob_pattern_test = os.path.join('dataset_preparation/', 'UCF_test','{classname}','*.avi')

    generator = SlidingFrameGenerator(
        sequence_stride = sequence_stride,
        frame_stride= frame_stride,
        rescale=1.0,
        classes=classes, 
        glob_pattern=glob_pattern_test,
        nb_frames=nb_frames,
        shuffle=False,
        batch_size=batch_size,
        target_shape=image_shape,
        nb_channel=3,
        transformation=test_datagen,
        use_frame_cache=False,
        preprocessing_type='MobileNet')

    return generator




