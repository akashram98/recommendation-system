import math
import time
from operator import add
import sys
# import numpy as np
from pyspark import SparkContext
import json
import os
import xgboost as xgb
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


def modelLayers(trainX, testX, trainY):
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
    r = [item for sublist in result for item in sublist]
    return r


def getPhoto(doc):
    item = doc[1]
    if item[1] is not None:
        return doc[0], item[1]
    else:
        return doc[0], 0


if __name__ == '__main__':

    startTime = time.time()

    # initialiseEnvironmentVariables()
    val1 = ""
    val2 = "yelp_val.csv"
    val3 = "output.csv"
    # val1 = sys.argv[1]
    # val2 = sys.argv[2]
    # val3 = sys.argv[3]

    sc = SparkContext('local[*]')
    sc.setLogLevel("FATAL")
    business = sc.textFile(val1 + 'business.json').map(lambda doc: json.loads(doc))

    distinctBusiness = business.map(lambda doc: doc['business_id']).distinct().sortBy(lambda doc: doc).zipWithIndex() \
        .map(lambda doc: {doc[0]: doc[1]}).flatMap(lambda doc: doc.items()).collectAsMap()
    distinctBusinessMax = max(distinctBusiness.values())

    businessStars = business.map(lambda doc: (doc['business_id'], doc['stars'])).collectAsMap()

    businessStarMax = max(businessStars.values())
    businessStarMin = min(businessStars.values())
    businessDiff = businessStarMax - businessStarMin

    # businessHours = business.map(lambda doc: (doc['business_id'], len(doc['hours']))).collectAsMap()
    # businessHoursMax = max(businessHours.values())
    # businessHoursMin = min(businessHours.values())
    # businessHoursDiff = businessHoursMax - businessHoursMin

    businessReviewCount = business.map(lambda doc: (doc['business_id'], doc['review_count'])).collectAsMap()
    businessReviewMax = max(businessReviewCount.values())
    businessReviewMin = min(businessReviewCount.values())
    businessReviewDiff = businessReviewMax - businessReviewMin

    useful = sc.textFile(val1 + 'user.json').map(lambda doc: json.loads(doc)) \
        .map(lambda doc: (doc['user_id'], doc['useful'] + doc['funny'] + doc['cool'])).collectAsMap()
    usefulMax = max(useful.values())
    usefulMin = min(useful.values())
    usefulDiff = usefulMax - usefulMin

    # print(business)
    review = sc.textFile(val1 + 'user.json').map(lambda doc: json.loads(doc)) \
        .map(lambda doc: (doc['user_id'], doc['average_stars'])) \
        .collectAsMap()
    # print(review)

    reviewMax = max(review.values())
    reviewMin = min(review.values())
    reviewDiff = reviewMax - reviewMin

    reviewCount = sc.textFile(val1 + 'user.json').map(lambda doc: json.loads(doc)) \
        .map(lambda doc: (doc['user_id'], doc['review_count'])) \
        .collectAsMap()

    reviewCountMax = max(reviewCount.values())
    reviewCountMin = min(reviewCount.values())
    reviewCountDiff = reviewCountMax - reviewCountMin

    # varReview = sc.textFile('review_train.json').map(lambda doc: json.loads(doc)) \
    #     .map(lambda doc: (doc['user_id'], doc['stars'] - review[doc['user_id']])) \
    #     .aggregateByKey((0, 0), lambda a, b: (a[0] + (b ** 2), a[1] + 1), lambda a, b: (a[0] + b[0], a[1] + b[1])) \
    #     .mapValues(lambda doc: doc[0] / (doc[1] - 1)).collectAsMap()

    # varReviewMax = max(varReview.values())
    # varReviewMin = min(varReview.values())
    # varReviewDiff = varReviewMax-varReviewMin

    photoReview = sc.textFile(val1 + 'review_train.json').map(lambda doc: json.loads(doc)).map(
        lambda doc: (doc['user_id'], 0))
    photoBusiness = business.map(lambda doc: (doc['business_id'], 0))
    photoCaption = sc.textFile(val1 + "photo.json").map(lambda doc: json.loads(doc)) \
        .map(lambda doc: (doc['business_id'], 1)).reduceByKey(add)
    photoInfo = photoBusiness.leftOuterJoin(photoCaption).map(lambda doc: getPhoto(doc)).collectAsMap()

    photoMax = max(photoInfo.values())
    photoMin = min(photoInfo.values())
    photoDiff = photoMax - photoMin
    # print(photoInfo)

    yelpSince = sc.textFile(val1 + 'user.json').map(lambda doc: json.loads(doc)) \
        .map(lambda doc: (doc['user_id'], int(doc['yelping_since'][:4]))).collectAsMap()

    yelpSinceMax = max(yelpSince.values())
    yelpSinceMin = min(yelpSince.values())
    yelpSinceDiff = yelpSinceMax - yelpSinceMin

    tipInfo = sc.textFile(val1 + 'tip.json').map(lambda doc: json.loads(doc)).persist()
    tipInfoUser = tipInfo.map(lambda doc: (doc['user_id'], 1)).reduceByKey(add)
    tipInfoUserFinal = photoReview.leftOuterJoin(tipInfoUser).map(lambda doc: getPhoto(doc)).collectAsMap()

    tipUserMax = max(tipInfoUserFinal.values())
    tipUserMin = min(tipInfoUserFinal.values())
    tipUserDiff = tipUserMax - tipUserMin

    tipInfoBusiness = tipInfo.map(lambda doc: (doc['business_id'], 1)).reduceByKey(add)
    tipInfoBusinessFinal = photoBusiness.leftOuterJoin(tipInfoBusiness).map(lambda doc: getPhoto(doc)).collectAsMap()

    tipBusinessMax = max(tipInfoBusinessFinal.values())
    tipBusinessMin = min(tipInfoBusinessFinal.values())
    tipBusinessDiff = tipBusinessMax - tipBusinessMin

    checkinFile1 = sc.textFile(val1 + "checkin.json").map(lambda doc: json.loads(doc)) \
        .map(lambda doc: (doc['business_id'], len(doc['time'])))

    checkinFile = photoBusiness.leftOuterJoin(checkinFile1).map(lambda doc: getPhoto(doc)).collectAsMap()

    checkinFileMax = max(checkinFile.values())
    checkinFileMin = min(checkinFile.values())
    checkinFileDiff = checkinFileMax - checkinFileMin

    # tip.json for users and business

    yelpTrain = sc.textFile(val1 + 'yelp_train.csv').map(lambda doc: doc.split(',')).filter(
        lambda doc: doc[0] != 'user_id') \
        .persist()

    yelpTrainX = yelpTrain.map(lambda doc: [
        # (distinctBusiness[doc[1]] / distinctBusinessMax),
        (review[doc[0]] - reviewMin) / reviewDiff,
        (businessStars[doc[1]] - businessStarMin) / businessDiff,
        (businessReviewCount[doc[1]] - businessReviewMin) / businessReviewDiff,
        (photoInfo[doc[1]] - photoMin) / photoDiff,
        (useful[doc[0]] - usefulMin) / usefulDiff,
        (reviewCount[doc[0]] - reviewCountMin) / reviewCountDiff,
        (yelpSince[doc[0]] - yelpSinceMin) / yelpSinceDiff,
        (tipInfoBusinessFinal[doc[1]] - tipBusinessMin) / tipBusinessDiff,
        (tipInfoUserFinal[doc[0]] - tipUserMin) / tipUserDiff,
        (checkinFile[doc[1]] - checkinFileMin) / checkinFileDiff
    ]).collect()

    yelpTrainY = yelpTrain.map(lambda doc: float(doc[2])).collect()

    # print(yelpTrainX)

    yelpVal = sc.textFile(val2).map(lambda doc: doc.split(',')).filter(lambda doc: doc[0] != 'user_id') \
        .map(lambda doc: [
        # (distinctBusiness[doc[1]] / distinctBusinessMax),
        (review[doc[0]] - reviewMin) / reviewDiff,
        (businessStars[doc[1]] - businessStarMin) / businessDiff,
        (businessReviewCount[doc[1]] - businessReviewMin) / businessReviewDiff,
        (photoInfo[doc[1]] - photoMin) / photoDiff,
        (useful[doc[0]] - usefulMin) / usefulDiff,
        (reviewCount[doc[0]] - reviewCountMin) / reviewCountDiff,
        (yelpSince[doc[0]] - yelpSinceMin) / yelpSinceDiff,
        (tipInfoBusinessFinal[doc[1]] - tipBusinessMin) / tipBusinessDiff,
        (tipInfoUserFinal[doc[0]] - tipUserMin) / tipUserDiff,
        (checkinFile[doc[1]] - checkinFileMin) / checkinFileDiff
    ]).collect()

    yelpTrainX = np.array(yelpTrainX)
    yelpTrainY = np.array(yelpTrainY)
    yelpVal = np.array(yelpVal)
    # print(yelpTrainY)
    res = modelLayers(yelpTrainX, yelpVal, yelpTrainY)
    result = list(res)
    # xgbRegression = xgb.XGBRegressor()

    # xgbRegression.fit(yelpTrainX, yelpTrainY)
    # result = xgbRegression.predict(yelpVal)
    # result = list(result)
    # print(result)
    # actualResult = sc.textFile('yelp_val.csv').map(lambda doc: doc.split(',')) \
    #     .filter(lambda doc: doc[0] != 'user_id').map(lambda doc: float(doc[2])).collect()
    # MSE = np.square(np.subtract(actualResult, result)).mean()
    #
    # RMSE = math.sqrt(MSE)
    # print("Root Mean Square Error:\n")
    # print(RMSE)

    v1 = sc.textFile(val2).map(lambda doc: doc.split(',')).filter(lambda doc: doc[0] != 'user_id').collect()

    file = open(val3, "w")
    file.write("user_id, business_id, prediction\n")
    for i, j in zip(result, v1):
        file.write(j[0] + "," + j[1] + "," + str(i))
        file.write("\n")

    print(time.time() - startTime)
    # yelpVal = sc.textFile('yelp_train.csv').map(lambda doc: doc.split(',')).collect()

    # print(yelpVal)
