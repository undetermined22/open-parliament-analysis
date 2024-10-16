# Databricks notebook source
# MAGIC %run ../includes/configuration

# COMMAND ----------

from pyspark.sql.window import Window
import pyspark.sql.functions as f

# COMMAND ----------

all_questions_df = spark.read\
    .parquet(f"{ingested_folder_path}/classified_questions")

# COMMAND ----------

deputy_dim_temp_df = all_questions_df\
    .withColumn('groupe', f.trim(f.regexp_replace(f.regexp_replace(f.col('groupe'), '[-]', ' '), '  ', ' ')))\
    .withColumn('groupe', f.regexp_replace(f.col('groupe'), 'وال', 'و ال'))\
    .withColumn('depute', f.explode(f.split(f.col('depute'), ',')))\
    .withColumn('depute', f.explode(f.split(f.col('depute'), '،')))\
    .withColumn('depute', f.trim(f.col('depute')))\
    .withColumn('depute', f.regexp_replace(f.col('depute'), 'بلافريج عمر', 'عمر بلافريج'))\
    .withColumn('depute', f.regexp_replace(f.col('depute'), '^فيقة لشرف$', 'شفيقة لشرف'))\
    .select(
        f.col('depute').alias('deputy_name'),
        f.col('groupe').alias('deputy_team'),
        f.col('legislation_id')
    )\
    .distinct()\
    .sort('legislation_id', 'deputy_team')\
    .withColumn('deputy_id', f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))\
    .select('deputy_id', 'deputy_name', 'deputy_team', 'legislation_id')

# COMMAND ----------

display(deputy_dim_temp_df)

# COMMAND ----------

# parties_groups_df = spark.read\
#     .parquet(f"{lookup_folder_path}/parties_groups")

# COMMAND ----------

# deputies_without_group_df = spark.read\
#     .parquet(f"{lookup_folder_path}/deputies_without_group")

# deputies_without_group_df = deputies_without_group_df.select([f.col(c).alias("wo_group_"+c) for c in deputies_without_group_df.columns])

# COMMAND ----------

deputies_parties_df = spark.read.parquet(f"{lookup_folder_path}/deputies_parties")

# COMMAND ----------

deputy_dim_temp_df = deputy_dim_temp_df\
    .join(deputies_parties_df,
        (
            (deputy_dim_temp_df['deputy_name'] == deputies_parties_df['depute']) &\
            (deputy_dim_temp_df['deputy_team'] == deputies_parties_df['groupe']) &\
            (deputy_dim_temp_df['legislation_id'] == deputies_parties_df['legislation_id']) 
        ),
        how="left"
    )\
    .withColumnRenamed('party_code', 'deputy_party_code')\
    .select('deputy_id', 'deputy_name', 'deputy_team', 'deputy_party_code', deputy_dim_temp_df['legislation_id'])

# COMMAND ----------

display(deputy_dim_temp_df)

# COMMAND ----------

deputy_party_dim_df = deputy_dim_temp_df\
    .select('deputy_party_code')\
    .distinct()\
    .sort('deputy_party_code')\
    .withColumn('deputy_party_id', f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))\
    .select('deputy_party_id', 'deputy_party_code')

deputy_party_dim_df.write.mode("overwrite").parquet(f"{processed_folder_path}/deputy_party_dim")

# COMMAND ----------

deputy_team_dim_df = deputy_dim_temp_df\
    .select(
        f.col('deputy_team').alias('deputy_team_name')
    )\
    .distinct()\
    .sort('deputy_team')\
    .withColumn('deputy_team_id', f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))\
    .select('deputy_team_id', 'deputy_team_name')

deputy_team_dim_df.write.mode("overwrite").parquet(f"{processed_folder_path}/deputy_team_dim")

# COMMAND ----------

deputy_dim_df = deputy_dim_temp_df\
    .join(
        deputy_team_dim_df,
        deputy_dim_temp_df['deputy_team'] == deputy_team_dim_df['deputy_team_name'],
        'left'
    )\
    .join(
        deputy_party_dim_df,
        deputy_dim_temp_df['deputy_party_code'] == deputy_party_dim_df['deputy_party_code'],
        'left'
    )\
    .select(
        'deputy_id',
        'deputy_name',
        'deputy_team_id',
        'deputy_party_id',
        'legislation_id'
    )\
    .sort('deputy_id')

deputy_dim_df.write.mode("overwrite").parquet(f"{processed_folder_path}/deputy_dim")

# COMMAND ----------

display(deputy_dim_df.count())

# COMMAND ----------

ministry_dim_df = all_questions_df\
    .select(
        f.col('ministry_classified').alias('ministry_name')
    )\
    .distinct()\
    .sort('ministry_name')\
    .withColumn('ministry_id', f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))\
    .select('ministry_id', 'ministry_name')

ministry_dim_df.write.mode("overwrite").parquet(f"{processed_folder_path}/ministry_dim")

# COMMAND ----------

session_dim_df = all_questions_df\
    .select(
        f.col('periode').alias('session_label')
    )\
    .distinct()\
    .withColumn('session_id', f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))\
    .select('session_id', 'session_label')

session_dim_df.write.mode("overwrite").parquet(f"{processed_folder_path}/session_dim")

# COMMAND ----------

question_fact_df = all_questions_df\
    .sort('date_depot')\
    .withColumn('question_id', f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))\
    .withColumn('depute', f.explode(f.split(f.col('depute'), ',')))\
    .select('question_id', *all_questions_df.columns)\
    .sort('question_id')

question_fact_df = question_fact_df\
    .join(deputy_dim_df, question_fact_df['depute'] == deputy_dim_df['deputy_name'])\
    .join(ministry_dim_df, question_fact_df['ministry_classified'] == ministry_dim_df['ministry_name'])\
    .join(session_dim_df, question_fact_df['periode'] == session_dim_df['session_label'])\
    .select(
        'question_id',
        f.col('numero').alias('question_number'),
        question_fact_df['legislation_id'],
        'session_id',
        'question_nature',
        'question_type',
        'deputy_id',
        'deputy_team_id',
        'deputy_party_id',
        'ministry_id',
        f.col('date_depot').alias('question_ask_date'),
        f.col('sujet').alias('question_subject'),
        f.col('question').alias('question_content'),
        f.col('sector').alias('question_sector'),
        f.isnotnull(f.col('date_reponse')).cast("integer").alias('question_is_replied'),
        f.col('date_reponse').alias('question_reply_date'),
        f.datediff('date_reponse', 'date_depot').alias('question_days_to_reply')
    )\
    .sort('question_id')

question_fact_df.write.mode("overwrite").parquet(f"{processed_folder_path}/question_fact")

# COMMAND ----------

display(question_fact_df.summary())

# COMMAND ----------

display(question_fact_df)
