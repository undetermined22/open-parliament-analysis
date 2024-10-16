# Databricks notebook source
# MAGIC %run ../includes/configuration

# COMMAND ----------

all_questions_df = spark.read\
    .parquet(f"{processed_folder_path}/question_fact")

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# COMMAND ----------


all_questions_df = all_questions_df\
    .withColumn('question_content', f.regexp_replace(f.col('question_content'), '[⸮–\(\)•\\\/\d":\.,\-؛،ـ_؟]', ' '))\
    .withColumn('question_content', f.regexp_replace(f.col('question_content'), ' وال', ' ال'))\
    .withColumn('question_content', f.regexp_replace(f.col('question_content'), ' ال ', ' '))\
    .withColumn('question_content', f.regexp_replace(f.col('question_content'), '\s+', ' '))\
    .withColumn('question_content', f.trim(f.col('question_content')))\
    .where(f.col('question_content').isNotNull())

display(all_questions_df)

# COMMAND ----------

arabic_stopwords = set(stopwords.words('arabic') + ['لذا', 'وفي', 'تم', 'وحيث', 'وذلك', 'والتي', 'وعليه', 'أنه', 'ظل', 'حول', 'عبر', 'عليها', 'يتم', 'لذلك', 'وكذا', 'لهذه'] + ["السيد", "الوزير", "المحترم", "نسائلكم", "الإجراءات", "التدابير", "الحكومة", "وزارتكم", "ستتخذونها", "خلال", "مستوى", "أسائلكم", "إطار", "المتخذة", "السيدة", "عدد", "العديد", "سنة", "الوزيرة", "طرف", "القطاع", "الأمر", "بشكل", "بسبب", "رقم", "مختلف", "المحترمة", "مجموعة", "فإننا", "عدم", "سيدي", "بهذه", "رغم", "لهذا", "عدة", "تعتزمون", "ستتخذها", "الأخيرة", "سواء", "وعن", "المتعلق", "حالة", "يعتبر", "بقبول", "سنوات", "رئيس", "كبيرة", "وقد", "الاستفادة", "السنة", "توفير", "أسباب", "•", "‏", "التعليمية", "المحلية", "العامة", "يلي", "تعلمون", "علما", "عبارات", "لأجل", "بتاريخ", "وأن", "تحقيق", "لفائدة", "تعتبر", "وعلى", "حصيلة", "سبق", "دعم", "ضمن", "المعنية", "الأطر", "الى", "ظروف", "الاجراءات", "كانت", "يعاني", "وتفضلوا", "فائق", "مهمة", "الخاص", "تنزيل", "كبيرا", "وخاصة", "وبعد" ,"لماذا", "التقدير", "الاحترام", "تحية", "تقدير", "واحترام", "يخفى", "عليكم", "نصره", "الله", "تنوون", "القيام", "اتخاذها", "بعين", "الاعتبار", "احترام" ,"قامت", "بهذا" ,"يتعلق", "تعرف", "بالنسبة"])

def preprocess_text(text):
  if not isinstance(text, str):
    return None
  # Tokenization
  tokens = [token for token in text.split(" ") if token.strip() != ""]

  # Remove stop words
  tokens = [word for word in tokens if word not in arabic_stopwords]
  
  return tokens

udf_preprocess_text = f.udf(preprocess_text, ArrayType(StringType()))

all_questions_df = all_questions_df.withColumn("tokens", udf_preprocess_text(f.col("question_content")))


# COMMAND ----------

def generate_ngrams(tokens, n):
  ngrams = []
  for i in range(len(tokens) - n + 1):
    for j in range(i, i+n):
        ngrams.append(tuple(tokens[i:j+1]))
  return ngrams

udf_ngrams = f.udf(generate_ngrams, ArrayType(ArrayType(StringType())))

all_questions_df = all_questions_df.withColumn("ngrams", udf_ngrams(f.col("tokens"), f.lit(3)))

# COMMAND ----------

display(all_questions_df)

# COMMAND ----------


all_questions_df = all_questions_df.withColumn("ngram", f.explode(f.col("ngrams"))).withColumn("ngram", f.array_join(f.col("ngram"), " "))
display(all_questions_df)

# COMMAND ----------

all_questions_df = all_questions_df.groupBy("ngram").count().orderBy("count", ascending=False)
display(all_questions_df.limit(100))
