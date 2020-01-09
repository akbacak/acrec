import os
import math
import cv2
import re
listing = os.listdir("/home/ubuntu/keras/enver/acrec/videos/")
count = 1
for file in listing:
    cap = cv2.VideoCapture("/home/ubuntu/keras/enver/acrec/videos/" + file)
    os.makedirs("/home/ubuntu/keras/enver/acrec/videos/Frames/" + file )
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    def getFrame(sec):
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = cap.read()
        if hasFrames:
            e=cv2.imwrite("/home/ubuntu/keras/enver/acrec/videos/Frames/"+ file +"/" + file + "_" +str(count)+".jpg", image) 
        return hasFrames

    sec = 0
    N=10    
    interwall = duration/N   #//it will capture image in each 0.5 second
    #print('fps = ' + str(fps))
    #print('number of frames = ' + str(frame_count))
    #print('duration (S) = ' + str(duration))
    #print('interwall = ' + str(interwall) )

    count = 1

    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + interwall
        success = getFrame(sec)
