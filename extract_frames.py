import os
import math
import cv2
import re
listing = os.listdir("/home/ubuntu/keras/enver/acrec/videos/")
count = 1
for file in listing:
    vidcap = cv2.VideoCapture("/home/ubuntu/keras/enver/acrec/videos/" + file)
    os.makedirs("/home/ubuntu/keras/enver/acrec/videos/Frames/" + file )
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite("/home/ubuntu/keras/enver/acrec/videos/Frames/"+ file +"/" + file + "_" +str(count)+".jpg", image) 
        return hasFrames
    sec = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)


