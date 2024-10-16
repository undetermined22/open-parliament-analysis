# Databricks notebook source
# MAGIC %run ../includes/configuration

# COMMAND ----------

from pyspark.sql.functions import col, lit, collect_set, countDistinct, array_join
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.feature import MinHashLSH

# COMMAND ----------

duplicate_questions_lhs_df = spark.read\
    .parquet(f"{processed_folder_path}/duplicate_questions_lhs")

# COMMAND ----------

all_questions_df = spark.read\
    .parquet(f"{processed_folder_path}/question_fact")

# COMMAND ----------

duplicate_ids = duplicate_questions_lhs_df.select(
    col("question_id").alias("question_id"),
    col("similar_question_id").alias("duplicate_question_id")
).union(
    duplicate_questions_lhs_df.select(
        col("similar_question_id").alias("question_id"),
        col("question_id").alias("duplicate_question_id")
    )
).distinct().groupBy("question_id").agg(
    collect_set("duplicate_question_id").alias("duplicate_question_ids"),
    countDistinct("duplicate_question_id").alias("duplicate_question_count")
).withColumn("duplicate_question_ids", array_join(col("duplicate_question_ids"), ", "))

# COMMAND ----------

all_questions_df = all_questions_df.join(
    duplicate_ids.withColumn("is_duplicate", lit(1)),
    on="question_id",
    how="left"
).fillna(0, subset=["is_duplicate"])\
.fillna(0, subset=["duplicate_question_count"])

# COMMAND ----------

all_questions_df.write.mode("overwrite").parquet(f"{processed_folder_path}/question_fact_duplicates")

# COMMAND ----------

display(all_questions_df.orderBy("duplicate_question_count", ascending=False))
