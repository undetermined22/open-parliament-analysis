# Databricks notebook source
# MAGIC %run ../includes/configuration

# COMMAND ----------

from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.feature import MinHashLSH

# COMMAND ----------

all_questions_df = spark.read\
    .parquet(f"{processed_folder_path}/question_fact")

# COMMAND ----------

tokenizer = Tokenizer(inputCol="question_content", outputCol="words")
words_df = tokenizer.transform(all_questions_df)

vectorizer = CountVectorizer(inputCol="words", outputCol="features", binary=True)
vectorized_model = vectorizer.fit(words_df)
vectorized_df = vectorized_model.transform(words_df)

# COMMAND ----------

mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
model = mh.fit(vectorized_df)

transformed_df = model.transform(vectorized_df)

# COMMAND ----------

similar_pairs = model.approxSimilarityJoin(transformed_df, transformed_df, 0.2, distCol="JaccardDistance")

# COMMAND ----------

similar_pairs_filtered = similar_pairs.filter(col("datasetA.question_id") != col("datasetB.question_id"))

# COMMAND ----------

similar_pairs_filtered.select("datasetA.*", col("datasetB.question_id").alias("similar_question_id"), "JaccardDistance").write.mode("overwrite").parquet(f"{processed_folder_path}/duplicate_questions_lhs")
