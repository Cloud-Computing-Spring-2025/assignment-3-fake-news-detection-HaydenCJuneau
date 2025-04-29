from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import to_timestamp, hour, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
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
    
    return df.select("id", "filtered_words", "features", "label_indexed")


def train_model(df: DataFrame) -> DataFrame:
    # Split into 80% Training and 20% Test
    train, test = df.randomSplit([0.8, 0.2])
    
    # Train Model with LR
    regression = LogisticRegression(featuresCol="features", labelCol="label_indexed")
    pipeline = Pipeline(stages=[regression]).fit(train)

    # Generate Predictions
    results = pipeline.transform(test)
    results.show()

    cols_d = ["id", "filtered_words", "label_indexed", "prediction"]

    return results.select([col(c).cast("string") for c in cols_d])


# Save result
processed_df = preprocess_data(df)
train_model(processed_df).coalesce(1).write.mode("overwrite").csv("/opt/bitnami/spark/Fake/output/task4", header=True)
 