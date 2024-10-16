# Databricks notebook source
raw_folder_path = 'abfss://raw@openparliamentdl.dfs.core.windows.net'
ingested_folder_path = 'abfss://ingested@openparliamentdl.dfs.core.windows.net'
processed_folder_path = 'abfss://processed@openparliamentdl.dfs.core.windows.net'
presentation_folder_path = 'abfss://presentation@openparliamentdl.dfs.core.windows.net'
lookup_folder_path = 'abfss://lookup@openparliamentdl.dfs.core.windows.net'

# COMMAND ----------

dl_account_key = dbutils.secrets.get(scope = 'openparliament-scope', key = 'openparliamentdl-account-key')
