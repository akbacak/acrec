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
import VideoGenerator


video = keras.Input(shape = (10,224,224,3), name='video')
cnn = InceptionV3(weights='imagenet',include_top=False, pooling='avg')
cnn.trainable =False



frame_features = layers.TimeDistributed(cnn)(video)
blstm_1 = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences= True))(frame_features)
blstm_2 = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences= False))(blstm_1)
Dense_2   = Dense(256, activation = 'sigmoid' )(blstm_2)
batchNorm = BatchNormalization()(Dense_2)
enver   = Dense(32, activation = 'sigmoid')(batchNorm)
batchNorm2= BatchNormalization()(enver)
Dense_3   = Dense(2, activation='sigmoid')(batchNorm2)
model = keras.models.Model(input = video , output = Dense_3)
print(model.summary())




self.generator = VideoGenerator(train_dir="/home/ubuntu/keras/enver/acrec/data/train/",
                                test_dir="/home/ubuntu/keras/enver/acrec/data/test/",
                                dims=(10, 224, 224, 3),
                                batch_size=16,
                                shuffle=True,
                                file_ext="np*" #".np*"
                                )

self.training_generator = self.generator.generate(train_or_test='train')
self.training_steps_per_epoch = len(self.generator.filenames_train) // self.batch_size
self.testing_generator = self.generator.generate(train_or_test="test")
self.testing_steps_per_epoch = len(self.generator.filenames_test) // self.batch_size

self.model.fit_generator(self.training_generator,
                         steps_per_epoch=self.training_steps_per_epoch,
                         epochs=epochs,
                         validation_data=self.testing_generator,
                         validation_steps=self.testing_steps_per_epoch)
