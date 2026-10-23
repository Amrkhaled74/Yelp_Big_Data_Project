# Set up all required functions
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path

# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")

# --- Bronze layer: load raw tip data and Silver dimensions ---
# Read raw tips data
df_read_tip_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_tip.json")

df_dim_user = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_user")

df_dim_business = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business")

df_dim_date = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_date")

df_dim_tips = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_tips")

# --- Silver: clean and normalize tip text ---
df_clean_text = df_read_tip_data.withColumn(
    "text_clean",
    F.lower(F.trim(F.regexp_replace("text", r"\s+", " ")))
).drop(F.col("text"))

# compresses elongated word characters (e.g., 'sooo' -> 'soo') to match NLP dictionary tokens
df_clean_text = df_clean_text.withColumn("text_clean", F.regexp_replace("text_clean", r"([a-z])\1{2,}", "$1$1"))
# collapses repeated punctuation marks (e.g., '!!!!' -> '!') into a single character
df_clean_text = df_clean_text.withColumn("text_clean", F.regexp_replace("text_clean", r"([!?.,*~-])\1{2,}", "$1"))

# --- Silver: resolve user dimension surrogate key ---
# inner join resolves the natural user identifier to its surrogate key and filters unmapped users
df_join_user_tips = df_clean_text.join(df_dim_user,on=df_clean_text["user_id"] == df_dim_user["User_id"],how="inner")\
                   .select(df_dim_user["User_pk_sk"],df_clean_text["*"]).drop(df_clean_text["user_id"])

# --- Silver: resolve business dimension surrogate key ---
# inner join resolves the natural business identifier to its surrogate key and filters unmapped businesses
df_join_user_tips_with_business = df_join_user_tips.join(df_dim_business , on = df_join_user_tips["business_id"] == df_dim_business["Business_id"] ,how="inner")\
                                  .select(df_join_user_tips["*"],df_dim_business["Business_pk_sk"]).drop(df_join_user_tips["business_id"])

# --- Silver: resolve date dimension surrogate key ---
# inner join maps tip timestamp string to the central date dimension surrogate key
df_join_date_with_dim_date = df_join_user_tips_with_business.join(df_dim_date , on = df_join_user_tips_with_business["date"] == df_dim_date["Date"] ,how="inner")\
                             .select(df_join_user_tips_with_business["*"] ,df_dim_date["Date_pk_sk"] ).drop(df_join_user_tips_with_business["date"])

# --- Silver: resolve tip text dimension surrogate key ---
# inner join maps normalized text to dim_tips surrogate key to enable out-of-fact text storage
df_get_text_pk  = df_join_date_with_dim_date.join(df_dim_tips , on = df_join_date_with_dim_date["text_clean"] == df_dim_tips["Text"] ,how="inner")\
                  .select(df_join_date_with_dim_date["*"] ,df_dim_tips["tips_pk_sk"] ).drop(df_join_date_with_dim_date["text_clean"])

# --- Silver: sanitize compliment metrics for fact table ---
# bounds validation ensures engagement metrics do not contain negative values
df_valid_review_number = df_get_text_pk.withColumn("clean_compliment_count",F.when( ( F.col("compliment_count") >= 0 ) ,F.col("compliment_count")).otherwise(0) ).drop(F.col("compliment_count"))

# --- Silver: write fact_tips to parquet storage ---
df_valid_review_number.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/fact_tips")