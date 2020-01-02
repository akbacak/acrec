import cv2
import keras
import numpy as np
from keras.applications import VGG16
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


listing = pd.read_csv('data2.csv')

NP = []

# for file in listing:
for file in listing.Video_ID:

    listing_2  = os.listdir("/home/ubuntu/Desktop/myDataset2/Frames/" + file + "/" )

    X = []
    for images in listing_2:
        image =  plt.imread("/home/ubuntu/Desktop/myDataset2/Frames/" + file + "/" + images )
        X.append (image)
    X = np.array(X)
    print(X.shape)


    NP.append(X)
    np.shape(NP)
    print(np.shape(NP))
NP = np.array(NP)
np.save(open("NP2.npy", 'w'), NP)



