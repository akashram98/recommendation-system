from pyspark import SparkContext
import json
import os
from operator import add

def initialiseEnvironmentVariables():
    os.environ['PYSPARK_PYTHON'] = "C:\\Users\\adiak\\AppData\\Local\\Programs\\Python\\Python38-32\\python.exe"
    os.environ['PYSPARK_DRIVER_PYTHON'] = "C:\\Users\\adiak\\AppData\\Local\\Programs\\Python\\Python38-32\\python.exe"

if __name__=="__main__":
    initialiseEnvironmentVariables()
    sc = SparkContext("local[*]")
    sc.textFile("sample.txt").map(lambda doc: doc.split(" "))\
        .flatMap(lambda doc: [(i, 1) for i in doc]).reduceByKey(add).foreach(lambda doc: print(doc))