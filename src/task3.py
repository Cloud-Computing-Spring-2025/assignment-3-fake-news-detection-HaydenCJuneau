from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import to_timestamp, hour, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml import Pipeline, Model

# Initialize Spark Session
spark: SparkSession = SparkSession.builder.appName("FakeNewsTask3").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load DataFrames
df = spark.read.option("header", True).csv("/opt/bitnami/spark/Fake/input/fake_news_sample.csv")


def preprocess_data(df: DataFrame) -> DataFrame:    
    # Convert 'text' column to lowercase and tokenize it
    tokenizer = Tokenizer(inputCol="text", outputCol="split_words")

    # Remove stop words
    remover = StopWordsRemover(inputCol="split_words", outputCol="filtered_words")
    
    # Hash tokenized text with HashingTF and IDF
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features")
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Use StringIndexer to convert labels into indexes
    label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed", handleInvalid="error")

    # Place TF-IDF features into one vector


    # Run Pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, label_indexer]).fit(df)
    df = pipeline.transform(df)

    # Export
    cols_d = ["id", "filtered_words", "features", "label_indexed"]

    df.show()
    
    return df.select([col(c).cast("string") for c in cols_d])


# Save result
preprocess_data(df).coalesce(1).write.mode("overwrite").csv("/opt/bitnami/spark/Fake/output/task3", header=True)
