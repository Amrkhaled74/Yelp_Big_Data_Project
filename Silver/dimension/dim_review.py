import os
import sys

from pyspark.sql import functions as F


from pathlib import Path

# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")

# --- Bronze layer: load raw review dataset ---
df_read_review_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_review.json")

df_review_text = df_read_review_data.select(F.col("review_id"),F.col("text"))

# --- Silver: clean and normalize review text ---
review = df_review_text.withColumn(
    "text_clean",
    F.lower(F.trim(F.regexp_replace("text", r"\s+", " ")))
).drop(F.col("text"))

# compresses elongated word characters (e.g., 'sooo' -> 'soo') to match NLP dictionary tokens
review = review.withColumn("text_clean", F.regexp_replace("text_clean", r"([a-z])\1{2,}", "$1$1"))
# collapses repeated punctuation marks (e.g., '!!!!' -> '!') into a single character
review = review.withColumn("text_clean", F.regexp_replace("text_clean", r"([!?.,*~-])\1{2,}", "$1"))

# removes reviews that consist only of whitespace, symbols, or emojis without alphanumeric content
df_clean = review.filter(F.col("text_clean").rlike(r"[a-zA-Z0-9]"))

# --- Silver: generate surrogate key for dim_review ---
# +1 avoids a 0-value surrogate key to follow standard dimensional modeling practice
df_surrogate_key = df_clean.withColumn("review_pk_sk", F.monotonically_increasing_id() + 1)

# --- Silver: project final review dimension schema ---
df_final = df_surrogate_key.select(F.col("review_pk_sk"),F.col("review_id"),F.col("text_clean").alias("review"))    

# --- Silver: write dim_review to parquet storage ---
df_final.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_review")