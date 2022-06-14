"""
Sliding frames
--------------

That module provides the SlidingFrameGenerator that is helpful
to get more sequence from one video file. The goal is to provide decayed
sequences for the same action.


"""

import os
import numpy as np
import cv2 as cv
from math import floor
from video_generator.generator_amir import VideoFrameGenerator

#from keras.utils import Sequence
from tensorflow.keras.utils import Sequence
#from keras.preprocessing.image import \
from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator, img_to_array
from psutil import virtual_memory
import tensorflow.keras.applications.resnet50 as resnet_utils
import tensorflow.keras.applications.mobilenet as mobilenet_utils

class SlidingFrameGenerator(VideoFrameGenerator):
    """
    SlidingFrameGenerator is useful to get several sequence of
    the same "action" by sliding the cursor of video. For example, with a
    video that have 60 frames using 30 frames per second, and if you want
    to pick 6 frames, the generator will return:

    - one sequence with frame ``[ 0,  5, 10, 15, 20, 25]``
    - then ``[ 1,  6, 11, 16, 21, 26])``
    - and so on to frame 30

    If you set `sequence_time` parameter, so the sequence will be reduce to
    the given time.

    params:

    - sequence_time: int seconds of the sequence to fetch, if None, the entire \
        vidoe time is used
    - sequence_stride: the step between two sequences, default=1\
        [a:b]->[a+sequence_stride:b++sequence_stride]->.....
    - frame_stride: the step between frames in a sequence, default=1\
        [a:frame_stride:b]
    from VideoFrameGenerator:

    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and validation
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that
        will be replaced by one of the class list
    """

    def __init__(self, *args, sequence_time: int = None, sequence_stride: int = 1, frame_stride: int = 1, preprocessing_type=None, **kwargs):
        super().__init__(no_epoch_at_init=True, *args, **kwargs)
        self.sequence_time = sequence_time
        self.sequence_stride = sequence_stride #amir
        self.frame_stride = frame_stride #amir
        self.sample_count = 0
        self.vid_info = []
        self.__frame_cache = {}
        self.__init_length()
        self.on_epoch_end()
        self.preprocessing = preprocessing_type

    def __init_length(self):
        count = 0
        print("Checking files to find possible sequences, please wait...")
        for filename in self.files:
            cap = cv.VideoCapture(filename)
            fps = cap.get(cv.CAP_PROP_FPS)
            frame_count = self.count_frames(cap, filename)
            cap.release()

            if self.sequence_time is not None:
                seqtime = int(fps*self.sequence_time)
            else:
                seqtime = int(frame_count)

            stop_at = int(seqtime - self.nbframe*self.frame_stride) #amir
            step = self.frame_stride #step = np.ceil(seqtime / self.nbframe).astype(np.int) - 1 #amir
            i = 0
            while i <= stop_at:#frame_count - stop_at: #amir
                self.vid_info.append({
                    'id': count,
                    'name': filename,
                    'frame_count': int(frame_count),
                    'frames': np.arange(i, i + seqtime)[::step][:self.nbframe],
                    'fps': fps,
                })
                count += 1
                i += self.sequence_stride #1 #amir

        print("For %d files, I found %d possible sequence samples" %
              (self.files_count, len(self.vid_info)))
        self.indexes = np.arange(len(self.vid_info))

    def on_epoch_end(self):
        # prepare transformation to avoid __getitem__ to reinitialize them
        if self.transformation is not None:
            self._random_trans = []
            for _ in range(len(self.vid_info)):
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.vid_info) / self.batch_size))

    def __getitem__(self, idx):
        classes = self.classes
        shape = self.target_shape
        nbframe = self.nbframe

        labels = []
        images = []

        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        transformation = None

        for i in indexes:
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            vid = self.vid_info[i]
            video = vid.get('name')
            frame_num = vid.get('frames') #amir
            classname = self._get_classname(video)

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            if self.use_frame_cache:
                if vid['id'] not in self.__frame_cache:
                    frames = self._get_frames_amir(video, frame_num, nbframe, shape)
                    if (virtual_memory()[1]/virtual_memory()[0])>0.3:  # add to cache, if there is space
                        self.__frame_cache[vid['id']] = frames
                else:
                    frames = self.__frame_cache[vid['id']]
            else:
                frames = self._get_frames_amir(video, frame_num, nbframe, shape)                


            # apply transformation
            if transformation is not None:
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]
            
            if self.preprocessing=='ResNet':
                frames = resnet_utils.preprocess_input(np.array(frames))
            
            if self.preprocessing=='MobileNet':
                frames = mobilenet_utils.preprocess_input(np.array(frames))
 
            # add the sequence in batch
            images.append(frames)
            labels.append(label)


        return np.array(images), np.array(labels)

    def _get_frames_amir(self, video, frame_num, nbframe, shape, force_no_headers=False): #amir
        cap = cv.VideoCapture(video)
        total_frames = self.count_frames(cap, video, force_no_headers)
        if total_frames % 2 != 0:
            total_frames += 1

        frames = []
        frame_i = 0

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            frame_i += 1
            if frame_i in frame_num:
                # resize
                frame = cv.resize(frame, shape)

                # use RGB or Grayscale ?
                if self.nb_channel == 3:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                else:
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                # to np
                frame = img_to_array(frame) * self.rescale

                # keep frame
                frames.append(frame)

            if len(frames) == nbframe:
                break

        cap.release()

        if not force_no_headers and len(frames) != nbframe:
            # There is a problem here
            # That means that frame count in header is wrong or broken,
            # so we need to force the full read of video to get the right
            # frame counter
            return self._get_frames(
                    video,
                    nbframe,
                    shape,
                    force_no_headers=True)

        if force_no_headers and len(frames) != nbframe:
            # and if we really couldn't find the real frame counter
            # so we return None. Sorry, nothing can be done...
            log.error("Frame count is not OK for video %s, "
                      "%d total, %d extracted" % (
                        video, total_frames, len(frames)))
            return None

        return np.array(frames)

    def get_validation_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            sequence_time=self.sequence_time,
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _validation_data=self.validation)

    def get_test_generator(self):
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            sequence_time=self.sequence_time,
            nb_frames=self.nbframe,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            glob_pattern=self.glob_pattern,
            _test_data=self.test)
