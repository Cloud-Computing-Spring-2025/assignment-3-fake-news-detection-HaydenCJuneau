from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import to_timestamp, hour

# Initialize Spark Session
spark: SparkSession = SparkSession.builder.appName("FakeNewsTask1").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load DataFrames
df = spark.read.option("header", True).csv("/opt/bitnami/spark/Fake/input/fake_news_sample.csv")


def explore(df: DataFrame) -> DataFrame:    
    # Create Temporary View
    df.createOrReplaceTempView("news_data")

    # Print first 5 rows
    spark.sql(
        "select * from news_data limit 5"
    ).show()

    # Count entries
    spark.sql(
        "select count(*) as story_count from news_data"
    ).show()
    
    # Find distict labels
    labels = spark.sql(
        "select distinct label from news_data"
    )
    labels.show()

    return labels


# Save result
explore(df).coalesce(1).write.mode("overwrite").csv("/opt/bitnami/spark/Fake/output/task1", header=True)
