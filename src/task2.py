from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import to_timestamp, hour, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Initialize Spark Session
spark: SparkSession = SparkSession.builder.appName("FakeNewsTask2").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load DataFrames
df = spark.read.option("header", True).csv("/opt/bitnami/spark/Fake/input/fake_news_sample.csv")


def tokenize(df: DataFrame) -> DataFrame:    
    # Convert 'text' column to lowercase and tokenize it
    tokenizer = Tokenizer(outputCol="split_words", inputCol="text")
    df = tokenizer.transform(df)

    remover = StopWordsRemover(outputCol="filtered_words", inputCol="split_words")
    df = remover.transform(df)

    cols_d = ["id", "title", "filtered_words", "label"]

    df.show()

    return df.select([col(c).cast("string") for c in cols_d])


# Save result
tokenize(df).coalesce(1).write.mode("overwrite").csv("/opt/bitnami/spark/Fake/output/task2", header=True)
