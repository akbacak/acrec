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
from keras.layers import SeparableConv2D,Conv1D,ConvLSTM2D
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


batch_size = 32
epochs = 120

begin  = Input(shape = ( X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
conv_1 = ConvLSTM2D(filters=5, kernel_size=(3, 3), activation='relu')(begin)
conv_2 = ConvLSTM2D(64, (3,3), activation='relu')(conv_1)
conv_3 = ConvLSTM2D(64, (3,3), activation='relu')(conv_2)
conv_4 = ConvLSTM2D(128, (3,3), activation='relu')(conv_3)
conv_5 = ConvLSTM2D(64, (3,3), activation='relu')(conv_4)
conv_6 = ConvLSTM2D(128, (3,3), activation='relu')(conv_5)
flatten= TimeDistributed(Flatten())(conv_6)
blstm_1=Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, input_shape=(X.shape[1], X.shape[2] * X.shape[3] * X.shape[4]), return_sequences =True))(flatten)
blstm_2=Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, input_shape=(X.shape[1], X.shape[2] * X.shape[3] * X.shape[4]), return_sequences =False))(blstm_1)
Dense_3 = Dense(4, activation = 'sigmoid')(blstm_2)
model = Model(input = begin, output = Dense_3)
print(model.summary())

from keras.optimizers import Adam
model.compile(loss = 'binary_crossentropy',  optimizer='adam') 
history = model.fit(X, Y, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1 )



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



score = model.evaluate(X_train, Y_train)
print(model.metrics_names)
print(score)

score = model.evaluate(X_valid, Y_valid)
print(model.metrics_names)
print(score)

score = model.evaluate(X, Y)
print(model.metrics_names)
print(score)
