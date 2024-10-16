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
import pyspark.sql.functions as f
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

result = data_clean.filter(f.col('words_clean').isNotNull())\
    .withColumn('word', f.explode(f.col('words_clean'))) \
    .groupBy('word') \
    .count()\
    .sort('count', ascending=False)

display(result.limit(200))

# COMMAND ----------


# tokenizer = Tokenizer(inputCol="sujet", outputCol="words_token")
tokenizer = RegexTokenizer(inputCol="question_subject", outputCol="words_token", pattern="[\s-,\"؛،\.ـ_:]")
tokenized = tokenizer.transform(
    written_questions_11_df
    .dropna(subset=["question_subject"])
    .withColumn('question_subject', f.regexp_replace(f.col('question_subject'), '[⸮\(\):\.,-؛،ـ_؟]', ' '))
    .withColumn('question_subject', f.regexp_replace(f.col('question_subject'), ' وال', ' ال'))
    .withColumn('question_subject', f.regexp_replace(f.col('question_subject'), ' ال ', ' '))
).select('question_id', 'words_token')

list_bis = ['لذا', 'وفي', 'تم', 'وحيث', 'وذلك', 'والتي', 'وعليه', 'أنه', 'ظل', 'حول', 'عبر', 'عليها', 'يتم', 'لذلك', 'وكذا', 'لهذه']
stopwordList = nltk.corpus.stopwords.words('arabic') + list_bis

remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean', stopWords=stopwordList)
data_clean = remover.transform(tokenized).select('question_id', 'words_clean')

result = data_clean.filter(f.col('words_clean').isNotNull())\
    .withColumn('word', f.explode(f.col('words_clean'))) \
    .groupBy('word') \
    .count()\
    .sort('count', ascending=False)

display(result.limit(100))
