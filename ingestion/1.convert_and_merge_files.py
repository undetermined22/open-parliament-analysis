# Databricks notebook source
# MAGIC %md
# MAGIC ## Packages, Configuration and Imports
# MAGIC

# COMMAND ----------

# MAGIC %run ../includes/configuration

# COMMAND ----------

import json
import pandas as pd
import csv
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge Sources

# COMMAND ----------

# MAGIC %md
# MAGIC ### Files Sources Configuration

# COMMAND ----------

f = open('files_list.json')

files_list = json.load(f)
f.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standarize and cleanup files

# COMMAND ----------

all_questions = None

for nature_legislation_key in files_list.keys():

    questions_df = pd.concat(
        map(lambda source: pd.read_excel(
                f"{raw_folder_path}/{source}",
                converters={'date_depot':str,'date_reponse':str},
                storage_options = {'account_key' : dl_account_key},
                engine='openpyxl'
            ),
            files_list[nature_legislation_key]['sources']
        ),
        ignore_index=True
    )

    # Drop num_seance column
    if 'num_seance' in questions_df.columns:
        questions_df.drop(labels=['num_seance'], axis=1, inplace=True)

    # Replace new line and tab space with space
    questions_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True, inplace=True)
    
    # Rename observation column to question
    if 'observation' in questions_df.columns:
        questions_df.rename(columns = {'observation':'question'}, inplace=True)

    # Drop rows where question is empty
    questions_df.dropna(axis=0, subset=['question'], inplace=True)

    # Drop rows with empty numero values
    questions_df.dropna(subset=['numero'], inplace=True)
    questions_df['numero'].astype(int)

    # Format dates
    questions_df.replace('-', None, inplace=True)
    questions_df['date_depot'] = pd.to_datetime(questions_df['date_depot'])
    questions_df['date_depot'] = questions_df['date_depot'].dt.strftime("%Y-%m-%d")
    questions_df['date_reponse'] = pd.to_datetime(questions_df['date_reponse'])
    questions_df['date_reponse'] = questions_df['date_reponse'].dt.strftime("%Y-%m-%d")

    # Add legislation and question type
    questions_df['question_type'] = files_list[nature_legislation_key]['type']
    questions_df['legislation_id'] = files_list[nature_legislation_key]['legislation']
    
    # Add question_nature column if not exist
    if 'type' not in questions_df.columns:
        questions_df['question_nature'] = np.nan
    else:
        questions_df.rename(columns = {'type':'question_nature'}, inplace=True)

    # Select Columns
    questions_df = questions_df[["numero", "legislation_id", "periode", "question_type", "question_nature", "depute", "groupe", "date_depot", "ministere", "sujet", "question", "date_reponse"]]

    # Merge into a single pyspark Dataframe
    if all_questions is None:
        all_questions = spark.createDataFrame(questions_df)
    else:
        all_questions = all_questions.union(spark.createDataFrame(questions_df))

# write PARQUET
all_questions.write.mode("overwrite").parquet(f"{ingested_folder_path}/questions")

# Display Summary
display(all_questions.summary())
