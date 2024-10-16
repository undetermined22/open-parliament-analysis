# Databricks notebook source
# MAGIC %run ../includes/configuration

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f

# COMMAND ----------

elected_pd_df = pd.read_excel(
    f"{raw_folder_path}/parlement-elus.xlsx",
    storage_options = {'account_key' : dl_account_key}
)

elected_df = spark.createDataFrame(elected_pd_df[['parlement', 'prenomNomAR', 'parti', 'groupeAR' ]])

elected_df = elected_df.where((elected_df.parlement == '2016-2021') | (elected_df.parlement == '2021-2026'))\
    .withColumn('parlement', f.when(f.col('parlement') == '2016-2021', f.lit(10)).otherwise(f.lit(11)))

# COMMAND ----------

parties = elected_df.select('parti')\
    .distinct()\
    .sort('parti')

parties.write.mode("overwrite").parquet(f"{lookup_folder_path}/parties")

# COMMAND ----------


parties_groups = elected_df.select('parlement', 'groupeAR', 'parti')\
    .where(elected_df.groupeAR != 'sans_groupe')\
    .withColumn('groupeAR', f.trim(f.regexp_replace(f.regexp_replace(f.col('groupeAR'), '[-]', ' '), '  ', ' ')))\
    .withColumn('groupeAR', f.regexp_replace(f.col('groupeAR'), 'وال', 'و ال'))\
    .distinct()\
    .sort('parti')

parties_groups_unique = parties_groups.join(
    parties_groups.select('groupeAR', 'parti').distinct().groupBy('groupeAR').count().where(f.col('count') == 1).select('groupeAR'),
    on='groupeAR'
)

parties_groups.write.mode("overwrite").parquet(f"{lookup_folder_path}/parties_groups")

# COMMAND ----------

all_questions_df = spark.read\
    .parquet(f"{ingested_folder_path}/questions")
    
all_questions_df = all_questions_df\
    .withColumn('groupe', f.trim(f.regexp_replace(f.regexp_replace(f.col('groupe'), '[-]', ' '), '  ', ' ')))\
    .withColumn('groupe', f.regexp_replace(f.col('groupe'), 'وال', 'و ال'))\
    .withColumn('depute', f.explode(f.split(f.col('depute'), ',')))\
    .withColumn('depute', f.explode(f.split(f.col('depute'), '،')))\
    .withColumn('depute', f.trim(f.col('depute')))\
    .withColumn('depute', f.regexp_replace(f.col('depute'), 'بلافريج عمر', 'عمر بلافريج'))\
    .withColumn('depute', f.regexp_replace(f.col('depute'), '^فيقة لشرف$', 'شفيقة لشرف'))\
    .select('depute', 'groupe', 'legislation_id')\
    .distinct()

# COMMAND ----------

all_questions_parties_df = all_questions_df.join(
    parties_groups_unique,
    on=(
        (f.col('groupe') == f.col('groupeAR')) &\
        (f.col('legislation_id') == f.col('parlement'))
    ),
    how='left'
).drop('groupeAR', 'parlement', 'count')

# COMMAND ----------

def add_comparison_name_field(dataframe, original_field, new_field):
    return dataframe\
        .withColumn(new_field, f.trim(f.regexp_replace(f.trim(f.col(original_field)), '[\s\t]+', ' ')))\
        .withColumn(new_field, f.regexp_replace(f.col(new_field), 'عبب?د ال', 'عبدال'))\
        .withColumn(new_field, f.regexp_replace(f.col(new_field), '[آأإ]', 'ا'))\
        .withColumn(new_field, f.regexp_replace(f.col(new_field), '[ئى]', 'ي'))\
        .withColumn(new_field, f.regexp_replace(f.col(new_field), 'ال', ''))\
        .withColumn(new_field, f.regexp_replace(f.col(new_field), 'مولاي', ''))\
        .withColumn(new_field, f.trim(f.array_join(f.sort_array(f.split(f.col(new_field), ' ')), ' ')))\
        .withColumn(new_field, f.trim(f.regexp_replace(f.trim(f.col(new_field)), '[\s\t]+', ' ')))

# COMMAND ----------

missing_questions_parties_df = add_comparison_name_field(
    all_questions_parties_df.where(f.col('parti').isNull()).drop('parti'),
    'depute',
    'comparison_name_questions'
)


elected_df = add_comparison_name_field(elected_df, 'prenomNomAR', 'comparison_name_elected')

# COMMAND ----------

joined_df_2 = missing_questions_parties_df\
    .drop("parti")\
    .crossJoin(
        elected_df
    )\
    .withColumn("is_similar", f.levenshtein(f.col("comparison_name_questions"), f.col("comparison_name_elected"), 1))\
    .where(
        (f.col("is_similar") != f.lit(-1)) &\
        ((f.col("groupe") == f.col("groupeAR")) | (f.col("groupeAR") == f.lit('sans_groupe'))) &\
        (f.col("legislation_id") == f.col("parlement"))
    )

# COMMAND ----------

display(joined_df_2.count())
display(missing_questions_parties_df.count())

# COMMAND ----------

deputies_parties_df = all_questions_parties_df.join(
    joined_df_2.withColumnRenamed('depute', 'deputeName').select('deputeName', 'parlement', 'groupeAR', 'parti'),
    on=(
        (all_questions_parties_df.parti.isNull()) &\
        (all_questions_parties_df.depute == f.col('deputeName')) &\
        (all_questions_parties_df.legislation_id == f.col('parlement')) &\
        ((all_questions_parties_df.groupe == f.col('groupeAR')) | (f.col('groupeAR') == f.lit('sans_groupe')))
    ),
    how='left')\
    .withColumn('party_code', f.when(
        all_questions_parties_df.parti.isNull(),
        joined_df_2.parti
    ).otherwise(all_questions_parties_df.parti))\
    .withColumn('party_code', f.when(
        f.col('depute') == f.lit('سعيدة زهير'),
        f.lit('UC')
    ).otherwise(f.col('party_code')))\
    .drop('parti', 'parlement', 'groupeAR', 'deputeName')

# COMMAND ----------

display(deputies_parties_df)

# COMMAND ----------

deputies_parties_df.write.mode("overwrite").parquet(f"{lookup_folder_path}/deputies_parties")
