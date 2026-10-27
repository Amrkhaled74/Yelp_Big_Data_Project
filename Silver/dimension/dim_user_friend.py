import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.functions import dense_rank, desc, when

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, CharType, ArrayType

from itertools import chain

import sys
from pathlib import Path

# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")

# --- Silver: load source data for surrogate resolution and friend arrays ---
df_read_dim_user = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_user")

df_read_user_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_user.json")

# alias natural key to prevent ambiguous column errors during downstream multi-join stages
df_dim_user_id_and_pk = df_read_dim_user.select(
    F.col("User_pk_sk"),
    F.col("User_id").alias("dim_user_id")
)

df_friend_user = df_read_user_data.select(
    F.col("friends"),
    F.col("user_id")
)

# --- Silver: normalize delimited friends list via UDF ---
@F.udf(returnType=ArrayType(StringType()))
def clean_friend_user(text):
    """
    Parse a comma-separated list of friend IDs into cleaned, whitespace-stripped string tokens.

    Provides a uniform list structure across users while substituting empty or null relationships
    with a sentinel token to avoid schema nullability issues during array transformations.
    Input is a raw comma-separated string; returns an array of trimmed friend ID strings.
    """
    if text is None:
       return ["has no friends"] 
     
    list_of_ids = text.split(",")
    list_of_ids_without_space = [i.strip() for i in list_of_ids if i is not None]         

    if len(list_of_ids_without_space) == 0:
        return ["has no friends"] 
    return list_of_ids_without_space

df_clean_friend_ids_col = df_friend_user.withColumn("friend_id", clean_friend_user(F.col("friends"))).drop(F.col("friends"))

# caching materialized array prevents re-evaluating Python UDF across subsequent multi-stage joins
df_clean_friend_ids_col.cache()

# count triggers cache materialization before pipeline execution continues
df_clean_friend_ids_col.count()

# --- Silver: flatten friend arrays and resolve surrogate keys ---
df_friends_exploded = df_clean_friend_ids_col.withColumn("friend_id_explode", F.explode(F.col("friend_id"))).drop(F.col("friend_id"))

# resolve friend natural ID to surrogate key; drops users with invalid IDs or the 'has no friends' sentinel
df_user_friend_joined = df_friends_exploded.join(
    df_dim_user_id_and_pk,
    on=df_friends_exploded["friend_id_explode"] == df_dim_user_id_and_pk["dim_user_id"],
    how="inner"
).drop(F.col("dim_user_id"))

# isolate and rename intermediate columns to avoid ambiguous column references on second dimension lookup
df_alias = df_user_friend_joined.select(
    F.col("user_id").alias("join_user_id"),
    F.col("friend_id_explode").alias("join_friend_id_explode"),
    F.col("User_pk_sk").alias("join_friend_User_pk_sk")
)

# resolve primary user natural ID to surrogate key
df_user_friends_with_keys = df_alias.join(
    df_dim_user_id_and_pk,
    on=df_alias["join_user_id"] == df_dim_user_id_and_pk["dim_user_id"],
    how="inner"
)

# --- Silver: assemble and persist bridge table ---
df_select_user_friend_pk = df_user_friends_with_keys.select(
    F.col("join_friend_User_pk_sk").alias("Friend_pk"),
    F.col("User_pk_sk").alias("User_pk")
)

#df_select_user_friend_pk.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_user_friend")