#In addition to v3, also creates a *.npy file for each video (video's frames), and save them to respective data's subfolders
import os
import math
import cv2
import re
import matplotlib.pyplot as plt
import numpy as np

listing = os.listdir("/home/ubuntu/keras/enver/acrec/videos/")
count = 1
for file in listing:
    train_val_dir  = os.listdir("/home/ubuntu/keras/enver/acrec/videos/" + file + "/" )   
    for file_2 in train_val_dir:
       class_dir = os.listdir("/home/ubuntu/keras/enver/acrec/videos/" + file + "/" + file_2 + "/") 
       os.makedirs("/home/ubuntu/keras/enver/acrec/data/" + file + "/" + file_2 )
       for videos in class_dir:
           cap = cv2.VideoCapture("/home/ubuntu/keras/enver/acrec/videos/" + file + "/" + file_2 + "/" + videos)
           os.makedirs("/home/ubuntu/keras/enver/acrec/Frames/" + file + "/" + file_2 + "/" + videos)
           fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
           frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
           duration = frame_count/fps
           def getFrame(sec):
               cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
               hasFrames,image = cap.read()
               if hasFrames:
                   e=cv2.imwrite("/home/ubuntu/keras/enver/acrec/Frames/"+ file +"/" + file_2 + "/" + videos +"/" + videos + "_" +str(count)+".jpg", image)
               return hasFrames
           sec = 0
           N = 10 # How many frames will be extracted
           interwall = duration/N   #//it will capture image in each 'interwall' second
           count = 1
           success = getFrame(sec)
           while success:
               count = count + 1
               sec = sec + interwall
               success = getFrame(sec)
           os.system('sh resize.sh')
           X = []
           frame_dir = os.listdir("/home/ubuntu/keras/enver/acrec/Frames/" + file +"/" + file_2 + "/" + videos +"/")
           for im in frame_dir:
               frame =  plt.imread("/home/ubuntu/keras/enver/acrec/Frames/" + file +"/" + file_2 + "/" + videos +"/" + im)
               X.append (frame)
           X = np.array(X)
           np.save(open("/home/ubuntu/keras/enver/acrec/data/" + file + "/" + file_2 + "/" + videos + ".npy", 'w'), X)


