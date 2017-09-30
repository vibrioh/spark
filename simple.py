import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import types


spark = SparkSession \
    .builder \
    .appName("simple") \
    .config("spark.som.config.option", "some-value") \
    .getOrCreate()

dataset1schema = types.StructType([
    types.StructField("mid", types.StringType(), False),
    types.StructField("title", types.StringType(), True),
    types.StructField("year", types.IntegerType(), True),
    types.StructField("score", types.FloatType(), True),
    types.StructField("reviews", types.IntegerType(), True)
])

dataset1 = spark.read.csv("/Users/vibrioh/local_projects/datasets/movie-dialogs-corpus/movies.csv", header='true', schema=dataset1schema)

dataset1.show()

dataset1.printSchema()

dataset1.count()

dataset1.select("title").show()

dataset1.filter(dataset1['year']>=1983).show()

dataset1.select(dataset1['title'], dataset1['score'] * 10).show()

