import keras
import tensorflow as tf
from keras import layers
from keras.applications import InceptionV3
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

Y = pd.read_csv(r'/home/ubuntu/keras/enver/dmlvh2/Y.csv') # Video files labels, one hot encoded
Y.shape
print(Y.shape[:])



listing = pd.read_csv('/home/ubuntu/keras/enver/dmlvh2/data.csv')

NP = []

# for file in listing:
for file in listing.Video_ID:

    listing_2  = os.listdir("/home/ubuntu/Desktop/myDataset/Frames/" + file + "/" )

    X = []
    for images in listing_2:
        image =  plt.imread("/home/ubuntu/Desktop/myDataset/Frames/" + file + "/" + images )
        X.append (image)
    X = np.array(X)

    NP.append(X)
    np.shape(NP)
X = np.array(NP)
X.shape
print(np.shape(X))
batch_size = 4
epochs = 32
hash_bits_1 = 256


video = keras.Input(shape = (X.shape[1], X.shape[2] , X.shape[3] , X.shape[4]), name='video')

cnn = InceptionV3(weights='imagenet',include_top=False, pooling='avg')
cnn.trainable =False

frame_features = layers.TimeDistributed(cnn)(video)

blstm_1     = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences= True))(frame_features)
blstm_2     = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences= False))(blstm_1)
HEL_2       = Dense(hash_bits_1, activation = 'sigmoid' )(blstm_2)
batchNorm   = BatchNormalization()(HEL_2)
Dense_2     = Dense(32, activation = 'sigmoid')(batchNorm)
batchNorm2  = BatchNormalization()(Dense_2)
Dense_3     = Dense(4, activation='sigmoid')(batchNorm2)
'''
Enc         = Dense(512, activation = 'sigmoid' )(frame_features)
HEL_1       = Dense(16, activation = 'sigmoid' )(Enc)
Dec         = Dense(512, activation = 'sigmoid' )(HEL_1)
Dec_end     = Dense(2048, activation = 'sigmoid' )(Dec)

model       = keras.models.Model(input = video , output = [Dense_3, Dec_end])
print(model.summary())
'''
model       = keras.models.Model(input = video , output = Dense_3)
print(model.summary())



from keras.optimizers import SGD
sgd = SGD(lr=0.002, decay = 1e-5, momentum=0.9, nesterov=True)

model.compile(loss = 'categorical_crossentropy',  optimizer=sgd, metrics=['accuracy'])
history = model.fit(X, Y, shuffle=True, batch_size=batch_size,epochs=epochs,verbose=1 )

model_json = model.to_json()
with open("models/acrec_v3_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/acrec_v3__weights.h5")


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




