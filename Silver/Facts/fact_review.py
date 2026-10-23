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

# --- Bronze layer: load raw reviews and dimension tables ---
df_read_review_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_review.json")

df_dim_user = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_user")

df_dim_business = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business")

df_dim_date = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_date")

df_dim_tips = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_tips")

df_dim_reviwe = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_review")

# --- Silver: sanitize metric columns and drop raw text ---
# drops raw text because free-text contents are normalized and stored separately in dim_review
df_clean_review = df_read_review_data.withColumn("clean_useful",F.when( ( F.col("useful").try_cast("int") >= 0 ) ,F.col("useful")).otherwise(0) )\
                                     .withColumn("clean_cool",F.when( ( F.col("cool").try_cast("int") >= 0 ) ,F.col("cool")).otherwise(0) )\
                                     .withColumn("clean_stars",F.when( ( F.col("stars").try_cast("int") >= 0 ) ,F.col("stars")).otherwise(0) )\
                                     .withColumn("clean_funny",F.when( ( F.col("funny").try_cast("int") >= 0 ) ,F.col("funny")).otherwise(0) )\
                                     .drop(F.col("cool"),F.col("funny"),F.col("stars"),F.col("useful"),F.col("text"))

# --- Silver: resolve user dimension surrogate key ---
# inner join resolves the natural user identifier to its surrogate key and drops orphan user records
df_join_user_review = df_clean_review.join(df_dim_user,on=df_clean_review["user_id"] == df_dim_user["User_id"],how="inner")\
                   .select(df_dim_user["User_pk_sk"],df_clean_review["*"]).drop(df_clean_review["user_id"])

# --- Silver: resolve business dimension surrogate key ---
# inner join resolves the natural business identifier to its surrogate key and drops orphan business records
df_join_user_tips_with_business = df_join_user_review.join(df_dim_business , on = df_join_user_review["business_id"] == df_dim_business["Business_id"] ,how="inner")\
                                  .select(df_dim_business["Business_pk_sk"],df_join_user_review["*"]).drop(df_join_user_review["business_id"])

# --- Silver: resolve date dimension surrogate key ---
# inner join maps the review timestamp to the centralized date dimension surrogate key
df_join_date_with_dim_date = df_join_user_tips_with_business.join(df_dim_date , on = df_join_user_tips_with_business["date"] == df_dim_date["Date"] ,how="inner")\
                             .select(df_dim_date["Date_pk_sk"],df_join_user_tips_with_business["*"]  ).drop(df_join_user_tips_with_business["date"])

# --- Silver: resolve review text dimension surrogate key ---
# inner join links the natural review_id to its text dimension surrogate key
df_get_review_pk  = df_join_date_with_dim_date.join(df_dim_reviwe , on = df_join_date_with_dim_date["review_id"] == df_dim_reviwe["review_id"] ,how="inner")\
                  .select(df_dim_reviwe["review_pk_sk"],df_join_date_with_dim_date["*"]  ).drop(df_join_date_with_dim_date["review_id"])

# --- Silver: write fact_review to parquet storage ---
df_get_review_pk.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/fact_review")