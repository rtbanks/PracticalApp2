import sys
import math as m
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.pipeline import *
from pyspark import SparkFiles
from pyspark.ml.feature import *


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
data = data.drop("DayOfWeek")
data = data.fillna(0)
data.show()

#Defining stages for string indexing and one hot encoding
#Define stage 1
stage_1 = StringIndexer(inputCol="UniqueCarrier", outputCol="UniqueCarrierIndex")
#Define stage 2
stage_2 = StringIndexer(inputCol="Origin", outputCol="OriginIndex")
#Define stage 3
stage_3 = StringIndexer(inputCol="Dest", outputCol="DestIndex")
#Define stage 4
stage_4 = OneHotEncoderEstimator(inputCols=[stage_1.getOutputCol(), stage_2.getOutputCol(),
                                            stage_3.getOutputCol()], outputCols=["UniqueCarrierEncoded",
                                            "OriginEncoded", "DestEncoded"])
#Define stage 5
stage_5 = VectorAssembler(inputCols= ['Year', 'Month', 'DayofMonth', 'UniqueCarrierEncoded',
                                      'OriginEncoded', 'DestEncoded', 'DepTime', 'DepDelay', 'TaxiOut',
                                      'CRSArrTime', 'CRSElapsedTime', 'Distance', 'DepSin', 'DepCos',
                                      'DayOfWeekSin', 'DayOfWeekCos'], outputCol='features')
#Define stage 6
stage_6 = LinearRegression(featuresCol='features', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)

#Organizing stages of pipeline
pipline = Pipeline(stages=[stage_1, stage_2, stage_3, stage_4, stage_5, stage_6])

#Partition data for testing and training
train_data, test_data = data.randomSplit([.8,.2], seed=1234)
train_data.show(3)
test_data.show(3)
#fitting the linear model to the pipeline
linearModel = pipline.fit(train_data)
lrm = linearModel.stages[-1]
#Printing statistics measures from the training data
#print("Coefficients:" + str(lrm.coefficients))
print("Intercept:" + str(lrm.intercept))

trainingSummary = lrm.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


#Predictions for linear model and printing out statistics metrics
predictions = linearModel.transform(test_data)
predictions.select("prediction", "ArrDelay", "features").show(20)
r2evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="r2")
rmseEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="rmse")
print("R Squared on test data = %g" % r2evaluator.evaluate(predictions))
print("Root Mean Squared Error on test data = %g" % rmseEvaluator.evaluate(predictions))

spark.stop()

