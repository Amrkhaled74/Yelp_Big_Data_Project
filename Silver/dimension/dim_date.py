# Set up all required functions
import os
import sys
from pyspark.sql import functions as F
from pathlib import Path

# --- Environment setup: configure paths and initialize Spark ---
# NOTE: purpose unclear
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")

# --- Bronze layer: load review and tip source datasets ---
df_read_review_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_review.json")

df_read_tip_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_tip.json")

df_review_data = df_read_review_data.select(F.col("date").alias("review_date"))

df_tip_data = df_read_tip_data.select(F.col("date").alias("tip_date"))

# --- Silver: unify and deduplicate timestamp keys across interactions ---
# full outer join aligns disparate timestamps from both interaction sources into a single column
result = df_review_data.join(df_tip_data, on=df_review_data["review_date"] == df_tip_data["tip_date"], how="full") \
    .withColumn("Date", F.coalesce(df_review_data["review_date"], df_tip_data["tip_date"])).drop(F.col("review_date"),F.col("tip_date"))

# distinct guarantees a unique set of date entries prior to assigning surrogate keys
df_join_tip_review_without_null = result.filter(F.col("Date").isNotNull()).distinct()

# --- Silver: decompose timestamp into calendar attributes ---
df = df_join_tip_review_without_null.withColumn("Year",F.year(F.col("Date")))\
.withColumn("Month",F.month(F.col("Date")))\
.withColumn("Day",F.day(F.col("Date")))

# --- Silver: generate surrogate key for dim_date ---
# +1 avoids a 0-value surrogate key to follow standard dimensional modeling practice
df_add_surrogate_key = df.withColumn("Date_pk_sk", F.monotonically_increasing_id() + 1)

# --- Silver: project final date dimension schema ---
df_final = df_add_surrogate_key.select(F.col("Date_pk_sk") , F.col("Date") , F.col("Year") , F.col("Month") , F.col("Day") )

# --- Silver: write dim_date to parquet storage ---
df_final.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_date")