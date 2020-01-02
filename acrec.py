import cv2
import glob
import numpy as np
import sys,os
import matplotlib
matplotlib.use('Agg')
import time
import os, os.path
import re,glob
import scipy.io
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.applications import VGG16
from keras.models import model_from_json
from keras.models import load_model



numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

X = []
files = sorted(glob.glob("/home/ubuntu/caffe/data/lamda_2/lamdaPics/*.jpg"),  key=numericalSort)

for myFile in files:
    image = cv2.imread (myFile)
    X.append (image)
X = np.array(X, dtype=np.float32)

Y = scipy.io.loadmat('/home/ubuntu/caffe/data/lamda/File_Lists/targets')
del Y['__version__']
del Y['__header__']
del Y['__globals__']
Y=list(Y.values())
Y = np.reshape(Y, (2000,5))

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2 ,random_state=13)
X_train = X_train.astype("float32")
X_valid = X_valid.astype("float32")
X_train /= 255
X_valid /= 255

image_size=256
batch_size = 16
epochs = 6
num_classes = Y_train.shape[1]

model = models.Sequential()
model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(image_size, image_size, 3)))
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.Conv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.Conv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='sigmoid'))


from keras.optimizers import SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'binary_crossentropy',  optimizer=sgd, metrics=['accuracy'])



train_dropout = model.fit(X_train, Y_train,
        shuffle=True,  batch_size=batch_size,epochs=epochs,verbose=1,
        validation_data=(X_valid, Y_valid) )



model_json = model.to_json()
with open("models/acrec_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/acrec_weights.h5")

