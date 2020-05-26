import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m
from pyspark.sql import functions as f
from pyspark.sql import udf
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import Row, Column
from pyspark import SparkFiles
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import DenseVector

spark = SparkSession.builder.appName("PracticalApplication").getOrCreate()

#load data from local file and input into application
inputData = sys.argv[1]  #"./files/2008.csv"
data = spark.read.load(inputData, format="csv", sep=",", inferSchema="true", header="true")

#Drop all forbidden variables from dataset
data = data.drop(*['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn',
                   'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
                   'SecurityDelay', 'LateAircraftDelay', 'CancellationCode', 'TailNum',
                   'Cancelled', 'CRSDepTime', 'FlightNum'])
#Check for null values in variables
data = data.dropna()
data_agg = data.agg(*[f.count(f.when(f.isnull(c), c)).alias(c) for c in data.columns])
data_agg.show()
#Check type of variables
data.printSchema()
data.show()

#UDF to converts times from military time to seconds using modulo
def get_seconds(value):
    if value > 0:
        x = value % 100
        y = ((value / 100) % 10)
        z = ((value / 1000) % 10)
        return (x * 60) + (y * 3600) + (z * 36000)
udfgetseconds = f.udf(get_seconds, IntegerType())

def get_seconds_mins(value):
    if value == 0:
        x = 0
    else:
        x = value
        return x * 60

udfgetsecondsmins = f.udf(get_seconds_mins, IntegerType())
#Convert selected columns from time to seconds and selected columns from minutes
#to seconds
data = data.withColumn("DepTime", udfgetseconds("DepTime"))\
                        .withColumn("CRSArrTime", udfgetseconds("CRSArrTime"))\
                        .withColumn("CRSElapsedTime", udfgetsecondsmins("CRSElapsedTime"))\
                        .withColumn("DepDelay", udfgetsecondsmins("DepDelay"))\
                        .withColumn("TaxiOut", udfgetsecondsmins("TaxiOut"))\
                        .withColumn("ArrDelay", udfgetsecondsmins("ArrDelay"))

#Converting data into sin and cos for cyclical transformation
#UDF to convert secs into sin
def get_sin(value):
    if value > 0:
        secs = 24*60*60
        x = m.sin(2*m.pi*value/secs)
        return x
udfgetsin = f.udf(get_sin, FloatType())

def get_sin_day(value):
    if value > 0:
        days = 7
        x = m.sin(2*m.pi*value/days)
        return x
udfgetsindays = f.udf(get_sin_day, FloatType())
#UDF to convert secs into cos
def get_cos(value):
    if value > 0:
        secs = 24*60*60
        x = m.cos(2*m.pi*value/secs)
        return x
udfgetcos = f.udf(get_cos, FloatType())

def get_cos_day(value):
    if value > 0:
        days = 7
        x = m.cos(2*m.pi*value/days)
        return x
udfgetcosdays = f.udf(get_cos_day, FloatType())


data.select(*[f.count(f.when(f.isnull(c), c)).alias(c) for c in data.columns]).show()

#Convert selected columns to sin and cos for cyclical transformation
data = data.withColumn("DepSin", udfgetsin("DepTime"))\
                        .withColumn("DepCos", udfgetcos("DepTime"))\
                        .withColumn("DayOfWeekSin", udfgetsindays("DayOfWeek"))\
                        .withColumn("DayOfWeekCos", udfgetcosdays("DayOfWeek"))
data = data.fillna(0)

data.show()








#import pandas as pd
#from pyspark.ml.feature import VectorAssembler
#from pyspark.ml.feature import StringIndexer


#data.select("ArrDelay", "DepDelay", "Distance", "TaxiOut").describe().show()
#Convert string values to index inorder to pass throught to vectors
#data_delay = data.select("ArrDelay", "Month", "DayOfWeek", "DepTime", "UniqueCarrier", "DepDelay", "Origin", "Dest", "Distance", "TaxiOut")
#SI_ArrDelay = StringIndexer(inputCol='ArrDelay', outputCol='ArrDelay_Index')
#SI_DepTime = StringIndexer(inputCol='DepTime', outputCol='DepTime_Index')
#SI_UniqueCarrier = StringIndexer(inputCol='UniqueCarrier', outputCol='UniqueCarrier_Index')
#SI_DepDelay = StringIndexer(inputCol='DepDelay', outputCol='DepDelay_Index')
#SI_Origin = StringIndexer(inputCol='Origin', outputCol='Origin_Index')
#SI_Dest = StringIndexer(inputCol='Dest', outputCol='Dest_Index')
#SI_TaxiOut = StringIndexer(inputCol='TaxiOut', outputCol='TaxiOut_Index')
#Transform data
#data_delay = SI_ArrDelay.fit(data_delay).transform(data_delay)
#data_delay = SI_DepTime.fit(data_delay).transform(data_delay)
#data_delay = SI_UniqueCarrier.fit(data_delay).transform(data_delay)
#data_delay = SI_DepDelay.fit(data_delay).transform(data_delay)
#data_delay = SI_Origin.fit(data_delay).transform(data_delay)
#data_delay = SI_Dest.fit(data_delay).transform(data_delay)
#data_delay = SI_TaxiOut.fit(data_delay).transform(data_delay)
#Select data that will be used for analysis
#data_delay.select('ArrDelay', 'ArrDelay_Index', 'DepTime', 'DepTime_Index', 'UniqueCarrier', 'UniqueCarrier_Index', 'DepDelay', 'DepDelay_Index', 'Origin', 'Origin_Index', 'Dest', 'Dest_Index', 'TaxiOut', 'TaxiOut_Index').show(10)
#Establish vectors for training and testing
#vectorAssembler = VectorAssembler(inputCols = ['Month', 'DayOfWeek', 'DepTime_Index', 'UniqueCarrier_Index', 'DepDelay_Index', 'Origin_Index', 'Dest_Index', 'Distance', 'TaxiOut_Index'], outputCol = 'features')
#data_v = vectorAssembler.transform(data_delay)
#data_v = data_v.select(['features', 'ArrDelay_Index'])
#data_v.show(3)
#Partition data for testing and training
#train_data, test_data = data_v.randomSplit([.8,.2], seed=1234)
#train_data.show(3)
#test_data.show(3)

#from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
#Set values for linear regression and print results
#lr = LinearRegression(featuresCol='features', labelCol='ArrDelay_Index', maxIter=10, regParam=0.3, elasticNetParam=0.8)
#lr_model = lr.fit(train_data)
#print("Coefficients:" + str(lr_model.coefficients))
#print("Intercept:" + str(lr_model.intercept))

#trainingSummary = lr_model.summary
#print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
#print("r2: %f" % trainingSummary.r2)

#train_data.describe().show()

#lr_predictions = lr_model.transform(test_data)
#lr_predictions.select("prediction", "ArrDelay_Index", "features").show(5)
#lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay_Index", metricName="r2")

#print("R Squared on test data = %g" % lr_evaluator.evaluate(lr_predictions))

#test_result = lr_model.evaluate(test_data)
#print("Root Mean Squared Error on test data = %g" % test_result.rootMeanSquaredError)

#print("numIterations: %d" % trainingSummary.totalIterations)
#print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
#trainingSummary.residuals.show()

#predictions = lr_model.transform(test_data)
#predictions.select("prediction", "ArrDelay_Index", "features").show()

spark.stop()

