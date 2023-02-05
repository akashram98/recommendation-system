import os

import numpy as np
from pyspark import SparkContext
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Dropout, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def initialiseEnvironmentVariables():
    os.environ['PYSPARK_PYTHON'] = "C:\\Users\\adiak\\AppData\\Local\\Programs\\Python\\Python38-32\\python.exe"
    os.environ['PYSPARK_DRIVER_PYTHON'] = "C:\\Users\\adiak\\AppData\\Local\\Programs\\Python\\Python38-32\\python.exe"

def modelLayers(trainX, testX, trainY, testY):
    model = Sequential()
    model.add(Dense(128, input_shape=(10,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    # opt = Adam(learning_rate=0.01)
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(trainX, trainY, epochs=1)
    res = model.predict(testX)

    result = list(res)
    print(np.sqrt(mean_squared_error(testY, res)))
    r = [item for sublist in result for item in sublist]
    # print(r)
# def modelLayers(trainX, testX, trainY, testY):
#     model = Sequential()
#     model.add(Conv1D(128, 1,activation="relu", input_shape=(10, 1)))
#     model.add(Flatten())
#     model.add(Dense(64, activation="relu"))
#     model.add(Dense(1))
#     model.compile(loss="mse", optimizer="adam")
#
#     model.fit(trainX, trainY, epochs=20)
#     res = model.predict(testX)
#     print(np.sqrt(mean_squared_error(testY, res)))

if __name__ == '__main__':
    # initialiseEnvironmentVariables()
    sc = SparkContext('local[*]')
    filepath = "/mnt/vocwork4/ddd_v1_w_DN9_1030793/asn825671_8/asn825672_1/resource/asnlib/publicdata/"
    trainX = sc.textFile('inputx.txt').map(lambda doc: doc.split(","))\
        .map(lambda doc: (float(doc[0]), float(doc[1]), float(doc[2]), float(doc[3]), float(doc[4]), float(doc[5]), float(doc[6]), float(doc[7]), float(doc[8]), float(doc[9])))\
        .collect()
    sc.textFile("inputx.txt").foreach(lambda doc: print(doc))

    testX = sc.textFile('outputx.txt').map(lambda doc: doc.strip("'").split(",")) \
        .map(lambda doc: (
    float(doc[0]), float(doc[1]), float(doc[2]), float(doc[3]), float(doc[4]), float(doc[5]), float(doc[6]),
    float(doc[7]), float(doc[8]), float(doc[9]))) \
        .collect()

    trainY = sc.textFile('inputy.txt').map(lambda doc: float(doc)).collect()

    testY = sc.textFile('yelp_val.csv').map(lambda doc: doc.split(",")).filter(lambda doc: doc[0] != 'user_id')\
        .map(lambda doc: float(doc[2])).collect()

    # trainX = np.array(trainX)
    # trainX = trainX.reshape(trainX.shape[0],trainX.shape[1],1)
    # testX = np.array(testX)
    # testX = testX.reshape(testX.shape[0],testX.shape[1],1)
    # trainY = np.array(trainY)
    # # trainY = trainY.reshape(trainY.shape[0],trainY.shape[1],1)
    # testY = np.array(testY)
    # print(trainX.shape[-1])
    modelLayers(trainX, testX, trainY, testY)
    # print(trainY)


