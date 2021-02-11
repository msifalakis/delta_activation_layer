
#%%
import os
from shutil import copy
import cv2

# %%
names = ("UCF_train", "UCF_test") 
for name in names:
    if not os.path.exists(name):
        os.makedirs(name)
        os.makedirs(os.path.join(name+"_frames"))

    classes_videos = []
    with open(name+'list01.txt', 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            line = p.split(' ')
            line = line[0].split('/')
            classes_videos.append(line)


    for i in classes_videos:
        if not os.path.exists(os.path.join(name,i[0])):
            os.makedirs(os.path.join(name,i[0]))
            os.makedirs(os.path.join(name+"_frames", i[0]))
            print("class "+ i[0])
        copy(os.path.join("UCF-101",i[0],i[1]), os.path.join(name,i[0],i[1]))

        vidcap = cv2.VideoCapture(os.path.join("UCF-101",i[0],i[1]))
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(name+"_frames", i[0], i[1] + "_" + str(count) + ".jpg"), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1

    print(name+" folder Copying and frame making Done!")







# # %%
# a = "detection_test" #"test"
# if not os.path.exists(a):
#     os.makedirs(a)
# classes_videos = []
# with open(a+'list01.txt', 'r') as txt:
#     paths = [read.strip() for read in txt.readlines()]
#     for p in paths:
#         line = p.split(' ')
#         line = line[0].split('/')
#         classes_videos.append(line)

# for i in classes_videos:
#     if not os.path.exists(os.path.join(a,i[0])):
#         os.makedirs(os.path.join(a,i[0]))
#     copy(os.path.join("UCF101",i[0],i[1]), os.path.join(a,i[0],i[1]))

# print(a+" folder Copying Done!")
# %%
