import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession \
    .builder \
    .appName("bitcoins") \
    .config("spark.som.config.option", "some-value") \
    .getOrCreate()

schema = StructType([
    StructField("index", IntegerType(), False),
    StructField("id-from", StringType(), True),
    StructField("id-to", StringType(), True),
    StructField("username_from", StringType(), True),
    StructField("username_to", StringType(), True),
    StructField("gender_from", StringType(), True),
    StructField("gender_to", StringType(), True),
    StructField("amount", FloatType(), True),
    StructField("month", IntegerType(), True),
    StructField("day", IntegerType(), True),
    StructField("year", IntegerType(), True)
])

bitcoins = spark.read.csv("/Users/vibrioh/Google\ Drive/Courses/E6893/project1/Spark_bitcoin/bitcoin_data.csv", header='True', schema=schema)

bitcoins.show()

bitcoins.printSchema()

bitcoins.write.parquet("hdfs://localhost:9000/user/vibrioh/parquet/bitcoins")



bitcoins.describe().show()

import matplotlib.pyplot as plt
import numpy as np
import pylab as P

from pyspark import Row

bitcoins.createOrReplaceGlobalTempView("table")


yearSelect = spark.sql("select year from global_temp.table where year>2010 and year<2017 order by year")


yearList = yearSelect.rdd.map(lambda  p: p.year).collect()

plt.hist(yearList)
plt.title("Trade Distribution of the Years\n")
plt.xlabel("Year")
plt.ylabel("Number of Trade")
plt.show(block=True)


bitcoins.groupBy("year").count().show()