#coding=utf-8
import cv2
import numpy as np
import sys,os
import time
import matplotlib
import scipy.io
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
import keras
from keras.models import Input,Model,InputLayer
from keras.layers import Dense, Dropout, Flatten, TimeDistributed
from keras.layers import SeparableConv2D,Conv1D,Conv2D,GlobalAveragePooling2D,MaxPooling2D
from keras.applications import VGG16
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize   # for resizing images
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
import keras.backend as K
from keras.layers.normalization import BatchNormalization
import math


Y = pd.read_csv(r'/home/ubuntu/keras/enver/dmlvh2/Y2.csv') # Video files labels, one hot encoded
Y.shape
print(Y.shape[:])



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

    NP.append(X)
    np.shape(NP)
X = np.array(NP)
#print(np.shape(X))
#X = X.reshape(X.shape[0], X.shape[1] , X.shape[2] * X.shape[3] * X.shape[4])
X.shape
print(np.shape(X))
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2 ,random_state=43)

batch_size = 2
epochs = 120


video = keras.Input(shape = (X.shape[1], X.shape[2] , X.shape[3] , X.shape[4]), name='video')
conv_1 = TimeDistributed(SeparableConv2D(32, (3,3), activation='relu'))(video)
conv_2 = TimeDistributed(SeparableConv2D(64, (3,3), activation='relu'))(conv_1)
conv_3 = TimeDistributed(SeparableConv2D(64, (3,3), activation='relu'))(conv_2)
conv_4 = TimeDistributed(SeparableConv2D(128, (3,3), activation='relu'))(conv_3)
conv_5 = TimeDistributed(SeparableConv2D(128, (3,3), activation='relu'))(conv_4)
conv_6 = TimeDistributed(SeparableConv2D(256, (3,3), activation='relu'))(conv_5)
flat   = TimeDistributed(GlobalAveragePooling2D())(conv_6)
blstm_1   = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences = True  ))(flat)
blstm_2   = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences = False ))(blstm_1)
Dense_2   = Dense(256, activation = 'sigmoid' )(blstm_2)
batchNorm = BatchNormalization()(Dense_2)
enver   = Dense(32, activation = 'sigmoid')(batchNorm)
batchNorm2= BatchNormalization()(enver)
Dense_3   = Dense(4, activation='sigmoid')(batchNorm2)
model = keras.models.Model(input = video , output = Dense_3)
print(model.summary())


from keras.optimizers import SGD
sgd = SGD(lr=0.001, decay = 1e-3, momentum=0.9, nesterov=True)

model.compile(loss = 'binary_crossentropy',  optimizer=sgd, metrics=['accuracy'])
history = model.fit(X_train, Y_train, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(X_valid, Y_valid) )


model_json = model.to_json()
with open("models/vacrec_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/vacrec__weights.h5")



params = {'legend.fontsize': 20,
          'legend.handlelength': 2,}
plt.rcParams.update(params)


plt.plot(history.history['acc'] , linewidth=3, color="green")
plt.plot(history.history['val_acc'], linewidth=3, color="blue")
plt.title('model accuracy' , fontsize=20)
plt.ylabel('accuracy' , fontsize=20)
plt.xlabel('epoch' , fontsize=20)
plt.legend( ['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], linewidth=3, color="green")
plt.plot(history.history['val_loss'],  linewidth=3, color="blue")
plt.title('model loss' , fontsize=20)
plt.ylabel('loss' , fontsize=20)
plt.xlabel('epoch' , fontsize=20)
plt.legend( ['train', 'validation'], loc='upper left')
plt.show()
