import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import types


spark = SparkSession \
    .builder \
    .appName("dataset2") \
    .config("spark.som.config.option", "some-value") \
    .getOrCreate()

dataset2schema = types.StructType([
    types.StructField("tid", types.StringType(), False),
    types.StructField("tag", types.StringType(), True)
])

dataset2tags = spark.read.csv("/Users/vibrioh/local_projects/datasets/stacksample/Tags.csv", header='False', schema=dataset2schema, mode='DROPMALFORMED')

dataset2tags.show()


dataset2tags.printSchema()

dataset2tags.count()

dataset2tags.group("tag").show()

dataset2tags.createOrReplaceTempView('tags')

sqlTags
#
# dataset2tags.filter(dataset2tags['year']>=1983).show()
#
# dataset2tags.select(dataset2tags['title'], dataset2tags['score'] * 10).show()

