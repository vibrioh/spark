import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark import SparkContext
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.readwriter import DataFrameReader



spark = SparkSession \
    .builder \
    .appName("simple") \
    .config("spark.sql.warehouse.dir", "file:${system:user.dir}/spark-warehouse") \
    .enableHiveSupport() \
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

dataset1.write.insertInto("dataset1", overwrite=True)


dataset1.write.saveAsTable('dataset1s', mode='overwrite')

sc = SparkContext(appName="simple")
sqlContext = SQLContext(sc)
hiveContext = HiveContext(sc)
hiveContext.setConf("hive.exec.dynamic.partition", "false")
hiveContext.setConf("spark.sql.orc.filterPushdown", "true")

data = hiveContext.sql("SELECT * from dataset1").write.format("orc").mode(
    "append").insertInto("dataset1")

sc.stop()

dataset1.write.parquet("hdfs://localhost:9000/user/vibrioh/simple")