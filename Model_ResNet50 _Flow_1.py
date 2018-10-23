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

data_generator_train = ImageDataGenerator(rescale=1./127.5)
data_generator_test = ImageDataGenerator()

def evaluate(e):
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
    batches = 0
    for x_batch, y_batch in data_generator_train.flow(testXDev, testYDev, batch_size=32):
        print('Epoch ', e, ' Start ', start, ' End ', end, ' Batch ', batches)
        loss_and_metrics = my_new_model.evaluate(x_batch, y_batch)
        file = open('C://Users//jimena//Desktop//TFM//resultados.txt','a') 
        strPrint = str('\n' + 'Epoch ' + str(e) + ' Loss and metrics ' + str(loss_and_metrics) + '\n')
        file.write(strPrint) 
        file.close()
        batches += 1
        if batches >= len(testXDev) / 32:
            # we need to break the loop by hand because the generator loops indefinitely
            break
        gc.collect()

imagesFolders = 'C://Users//jimena//Desktop//TFM//Dataset_Resize//train//'
epochs = 10
# Get filenames
filenames = os.listdir(imagesFolders)
filenames = [f for f in filenames]
# Before splits shuffles elements with a fixed seed so that the split is reproducible
np.random.seed(230)
filenames.sort()
np.random.shuffle(filenames)
whileSize = 9600
for e in range(epochs):
    print('Epoch', e)
    start = 0
    end = start + whileSize
    while start < len(filenames) - 1:
        print('Epoch ', e, ' Start ', start, ' End ', end)
        trainX = np.zeros(shape=(end - start,512,512,3), dtype=np.float16)
        labels = np.zeros(shape=(end - start), dtype=np.int)
        cont = 0
        for i in range(start,end):            
            img = Image.open(imagesFolders + filenames[i])
            img = np.array(img)
            trainX[cont] = img
            labels[cont] = filenames[i].split('_')[-2]    
            cont += 1
            gc.collect()
        trainY = to_categorical(labels, num_classes=17)
        start = end
        end = start + whileSize
        if end > len(filenames):
            end = len(filenames)
        batches = 0
        for x_batch, y_batch in data_generator_train.flow(trainX, trainY, batch_size=32):
            print(' Batch ', batches)
            my_new_model.train_on_batch(x_batch, y_batch)
            batches += 1
            if batches >= len(trainX) / 32:
            # we need to break the loop by hand because the generator loops indefinitely
                break
        gc.collect() 
    evaluate(e)

K.clear_session()