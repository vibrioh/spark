import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import Row

spark = SparkSession \
    .builder \
    .appName("airlines") \
    .config("spark.som.config.option", "some-value") \
    .getOrCreate()

schema = StructType([
    StructField("Year", IntegerType(), False),
    StructField("Month", IntegerType(), True),
    StructField("DayofMonth", IntegerType(), True),
    StructField("DayOfWeek", IntegerType(), True),
    StructField("DepTime", IntegerType(), True),
    StructField("CRSDepTime", IntegerType(), True),
    StructField("ArrTime", IntegerType(), True),
    StructField("CRSArrTime", IntegerType(), True),
    StructField("UniqueCarrier", StringType(), True),
    StructField("FlightNum", IntegerType(), True),
    StructField("TailNum", StringType(), True),
    StructField("ActualElapsedTime", IntegerType(), True),
    StructField("CRSElapsedTime", IntegerType(), True),
    StructField("AirTime", IntegerType(), True),
    StructField("ArrDelay", IntegerType(), True),
    StructField("DepDelay", IntegerType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("Distance", IntegerType(), True),
    StructField("TaxiIn", IntegerType(), True),
    StructField("TaxiOut", IntegerType(), True),
    StructField("Cancelled", IntegerType(), True),
    StructField("CancellationCode", StringType(), True),
    StructField("Diverted", IntegerType(), True),
    StructField("CarrierDelay", StringType(), True),
    StructField("WeatherDelay", StringType(), True),
    StructField("NASDelay", StringType(), True),
    StructField("SecurityDelay", StringType(), True),
    StructField("LateAircraftDelay", StringType(), True)
])

airlines = spark.read.csv("/Users/vibrioh/Downloads/*.csv.bz2", header='True', schema=schema)


airlines.show()

airlines.printSchema()

airlines.filter(airlines['Year']<2007).show()

airlines.write.parquet("hdfs://localhost:9000/user/vibrioh/airlines")

ArrDelay = spark.sql("select ArrDelay from airlines")
ArrDelay.describe().show()

DepDelay = spark.sql("select DepDelay from airlines")
DepDelay.describe().show()


