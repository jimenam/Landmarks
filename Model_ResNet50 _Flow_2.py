# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:14:42 2018

@author: jimena
"""

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf
import random as rn
from PIL import Image
import gc

num_classes = 17
image_size = 512

# The below is necessary in Python 3.2.3 onwards to have reproducible behavior for certain hash-based operations.

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread. Multiple threads are a potential source of non-reproducible results.
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

session_conf.gpu_options.allow_growth = True

from keras import backend as K

# The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

data_generator_train = ImageDataGenerator()
data_generator_dev = ImageDataGenerator()

tbCallBack =  TensorBoard(log_dir='/Tensorboard/Graph_ResNet50_12', histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

imagesFoldersDev = 'C://Users//jimena//Desktop//TFM//Dataset_Resize//dev//'
filenamesDev = os.listdir(imagesFoldersDev)
filenamesDev = [f for f in filenamesDev]
np.random.seed(230)
filenamesDev.sort()
np.random.shuffle(filenamesDev)
testXDev = np.zeros(shape=(18473,512,512,3), dtype=np.float)
labelsDev = np.zeros(shape=(18473), dtype=np.int)
contDev = 0
for i in range(0,18472):            
    img = Image.open(imagesFoldersDev + filenamesDev[i])
    img = np.array(img)
    testXDev[contDev] = img
    labelsDev[contDev] = filenamesDev[i].split('_')[-2]         
    contDev += 1
    gc.collect()
testYDev = to_categorical(labelsDev, num_classes=17)
    
imagesFoldersTrain = 'C://Users//jimena//Desktop//TFM//Dataset_Resize//train//'
# Get filenames
filenamesTrain = os.listdir(imagesFoldersTrain)
filenamesTrain = [f for f in filenamesTrain]
np.random.seed(230)
filenamesTrain.sort()
np.random.shuffle(filenamesTrain)
trainX = np.zeros(shape=(147785,512,512,3), dtype=np.float)
labels = np.zeros(shape=(147785), dtype=np.int)
cont = 0
for i in range(0,147784):            
    img = Image.open(imagesFoldersTrain + filenamesTrain[i])
    img = np.array(img)
    trainX[cont] = img
    labels[cont] = filenamesTrain[i].split('_')[-2]    
    cont += 1
    gc.collect()
trainY = to_categorical(labels, num_classes=17)

my_new_model.fit_generator(
        data_generator_train.flow(trainX, trainY, batch_size=32),
        epochs=10,
        callbacks=[tbCallBack, early_stopping],
        validation_data=data_generator_dev.flow(testXDev, testYDev, batch_size=32))

K.clear_session()