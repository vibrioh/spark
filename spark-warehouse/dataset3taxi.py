import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import types


spark = SparkSession \
    .builder \
    .appName("dataset3") \
    .config("spark.som.config.option", "some-value") \
    .getOrCreate()

dataset3schema = types.StructType([
    types.StructField("trid", types.StringType(), False),
    types.StructField("VendorID", types.IntegerType(), True),
    types.StructField("tpep_pickup_datetime", types.StringType(), True),
    types.StructField("tpep_dropoff_datetime", types.StringType(), True),
    types.StructField("passenger_count", types.IntegerType(), True),
    types.StructField("trip_distance", types.FloatType(), True),
    types.StructField("RatecodeID", types.IntegerType(), True),
    types.StructField("store_and_fwd_flag", types.StringType(), True),
    types.StructField("PULocationID", types.IntegerType(), True),
    types.StructField("DOLocationID", types.IntegerType(), True),
    types.StructField("payment_type", types.IntegerType(), True),
    types.StructField("fare_amount", types.FloatType(), True),
    types.StructField("extra", types.FloatType(), True),
    types.StructField("mta_tax", types.FloatType(), True),
    types.StructField("tip_amount", types.FloatType(), True),
    types.StructField("tolls_amount", types.FloatType(), True),
    types.StructField("improvement_surcharge", types.FloatType(), True),
    types.StructField("total_amount", types.FloatType(), True)
])

dataset3taxi = spark.read.csv("/Users/vibrioh/local_projects/datasets/tripdata.csv", header='True', schema=dataset3schema)

dataset3taxi.show()

dataset3taxi.printSchema()

dataset3taxi.count()

dataset3taxi.select("VendorID").show()

dataset3taxi.filter(dataset3taxi['total_amount']>=200).show()

dataset3taxi.select(dataset3taxi['trid']).filter(dataset3taxi['passenger_count']>=6).show()