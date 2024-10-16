# Databricks notebook source
# MAGIC %run ../includes/configuration

# COMMAND ----------

written_questions_11_df = spark.read\
    .option("header", True)\
    .parquet(f"{processed_folder_path}/question_fact")

# COMMAND ----------

from operator import add
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
import pyspark.sql.functions as f
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import SparseVector
import nltk
nltk.download("stopwords")

# COMMAND ----------


tokenizer = RegexTokenizer(inputCol="question_content", outputCol="words_token", pattern="[\s-,\"؛،\.ـ_:]")
tokenized = tokenizer.transform(
    written_questions_11_df
    .dropna(subset=["question_content"])
    .withColumn('question_content', f.regexp_replace(f.col('question_content'), '[⸮\(\):\.,-؛،ـ_؟]', ' '))
    .withColumn('question_content', f.regexp_replace(f.col('question_content'), ' وال', ' ال'))
    .withColumn('question_content', f.regexp_replace(f.col('question_content'), ' ال ', ' '))
).select('question_id', 'words_token')

stopwordList = nltk.corpus.stopwords.words('arabic') + ['لذا', 'وفي', 'تم', 'وحيث', 'وذلك', 'والتي', 'وعليه', 'أنه', 'ظل', 'حول', 'عبر', 'عليها', 'يتم', 'لذلك', 'وكذا', 'لهذه']

remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean', stopWords=stopwordList)
data_clean = remover.transform(tokenized).select('question_id', 'words_clean')

hashingTF = HashingTF(inputCol="words_clean", outputCol="rawFeatures", numFeatures=10000)
tfData = hashingTF.transform(data_clean)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tfData)
tfidfData = idfModel.transform(tfData)

# COMMAND ----------

display(tfidfData)

# COMMAND ----------

display(tfidfData.schema)

# COMMAND ----------

from pyspark.ml.linalg import SparseVector
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, FloatType

# Define schema for UDF output
schema = ArrayType(StructType([
    StructField("term_index", IntegerType(), False),
    StructField("score", FloatType(), False)
]))

# Define UDF function to handle SparseVector
def extract_terms(features):
    terms = []
    if isinstance(features, SparseVector):
        indices = features.indices.tolist()  # Convert to list for UDF compatibility
        values = features.values.tolist()    # Convert to list for UDF compatibility
        for i in range(len(indices)):
            terms.append((indices[i], values[i]))
    return terms

# Register the UDF
extract_terms_udf = udf(extract_terms, schema)

# Apply UDF to extract terms
tfidf_terms = tfidfData.withColumn("terms", extract_terms_udf(col("features")))

# Explode the terms into rows
from pyspark.sql.functions import explode

exploded_tfidf = tfidf_terms.select(explode(col("terms")).alias("term_score"))
term_score_df = exploded_tfidf.select(col("term_score.term_index").alias("term_index"), col("term_score.score").alias("score"))

# Collect term scores to driver node
term_score_rdd = term_score_df.rdd
term_score_list = term_score_rdd.collect()


# COMMAND ----------

from pyspark.ml.linalg import SparseVector
from pyspark.sql.functions import udf, col
from pyspark.sql.types import MapType, StringType, FloatType

def extract_word_scores(features):
    word_scores = {}
    if isinstance(features, SparseVector):
        indices = features.indices
        values = features.values
        for i in range(len(indices)):
            word_scores[indices[i]] = values[i]
    return word_scores

extract_word_scores_udf = udf(extract_word_scores, MapType(StringType(), FloatType()))
word_scores_df = tfidfData.withColumn("word_scores", extract_word_scores_udf(col("features")))

# Explode word scores into rows
from pyspark.sql.functions import explode

exploded_word_scores = word_scores_df.select(explode(col("word_scores")).alias("word", "score"))

# Aggregate and filter word scores
aggregated_word_scores = exploded_word_scores.groupBy("word").sum("score").orderBy("sum(score)", ascending=False)


# COMMAND ----------


word_scores = aggregated_word_scores.rdd.map(lambda row: (row["word"], row["sum(score)"])).collect()

# Convert to dictionary
word_score_dict = {word: score for word, score in word_scores}

# COMMAND ----------

display(word_score_dict)
