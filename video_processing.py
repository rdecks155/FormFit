# from PIL import Image
# import csv
# im = Image.open('./bar_closeup.jpg', 'r')

import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
#%matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images

count = 0
videoFile = "./videos/originals/video5.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
#frameRate = cap.get(5) #frame rate


# video3: frames = [14.7,16.5,18,19.9,21.6,23.5,25.3,26.8,28.5,30.4,32.7,35.3,37.6, 41, 43, 44.9, 46.8, 48.8, 50.6, 55.1]
# video4: frames = [5.4, 7.2, 8.8, 10.1, 13.1, 14.7, 16.4,17.9, 19.4, 21.2, 23.2, 24.8, 26.5, 28.2, 29.8, 31.4, 33, 34.5, 36]
frames = [5.1, 13, 16.3, 19.2, 22.3, 25.4, 28.4, 31.5, 34.7, 37.6, 40.6, 43.7, 50.2, 52.5, 55.5, 58.2, 60.9, 64.6, 68, 70.2, 73.1, 76.6, 79.8, 83.8, 87.1, 89.3, 91.6]
for time in frames:
    for i in range(3):
        if i == 0:
            time_milliseconds = (time - .1) * 1000
        elif i == 1:
            time_milliseconds = time * 1000
        else:
            time_milliseconds = (time + .1) * 1000
            
        cap.set(cv2.CAP_PROP_POS_MSEC,time_milliseconds)
        success, image = cap.read()
        if success:
            filename = "stage3_frame%d.jpg" % count;count+=1
            cv2.imwrite('./videos/bar_frames/stage3/' + filename, image)
cap.release()
