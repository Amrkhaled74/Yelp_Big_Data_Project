import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.functions import dense_rank, desc, when

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, CharType, ArrayType

from itertools import chain

import pandas as pd
import probablepeople as pp

import sys
from pathlib import Path

# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")

# --- Bronze layer: load raw user data ---
df_read_user_data = spark.read.parquet("D:\\A_Data_engineer\\Project_Yelp\\store_dimesnion_user\\Bronze_Store\\user")

# --- Silver: filter invalid characters from user names ---
df_user_remove_junk_symbols = df_read_user_data.filter(
    # Unicode ranges cover CJK Unified Ideographs, Hiragana, and Katakana
    (df_read_user_data["name"].rlike(r"[a-zA-Z]") | df_read_user_data["name"].rlike(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]"))
    &
    (~df_read_user_data["name"].rlike(r"[0-9@#$%^&*()_+=\[\]{};:\"\\|<>/?]"))
)

# --- Silver: extract distinct names for NER processing ---
names_df = df_read_user_data.select("name").filter(
        (F.col("name").rlike(r"[a-zA-Z]") | F.col("name").rlike(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]"))
        & ~F.col("name").rlike(r"[0-9@#$%^&*()_+=\[\]{};:\"\\|<>/?]")
    )\
    .distinct()

# checkpoint distinct names to disk: avoids recomputing expensive regex and distinct during driver collect
names_df.write.mode("overwrite").parquet("D:/spark_tmp/distinct_names")

# pulling distinct names to the driver: probablepeople runs only in single-threaded Python
names = pd.read_parquet("D:/spark_tmp/distinct_names")["name"].tolist()

list_names = [pp.parse(n) for n in names]

# --- Silver: filter corporate entity tags from parsed names ---
unauthorized_username = {
    "CorporationNameAndCompany",
    "CorporationName",
    "CorporationNameOrganization",
    "CorporationLegalType",
    "CorporationCommitteeType",
    "CorporationNameBranchType",
    "CorporationNameBranchIdentifier"
}

valid_username = []

for row in list_names:
    username = ""
    count = 0
    
    for name, username_type in row:
        if username_type not in unauthorized_username:
            username = username + "" + name
            count = count + 1

    valid_username.append(username)

# broadcast filtering against valid names list collected on driver
df_valid_username = df_user_remove_junk_symbols.filter(
    df_user_remove_junk_symbols["name"].isin(valid_username)
)

# --- Silver: validate and standardize signup date ---
df_valid_date = df_valid_username.filter(
    # Yelp was founded in October 2004; dates prior or in the future represent bad source data
    F.to_date(F.col("yelping_since")).between("2004-10-01", F.current_date())
).withColumn(
    "Join_Date", F.to_date(F.col("yelping_since"))
).drop(
    F.col("yelping_since")
)

# --- Silver: sanitize metric and engagement count columns ---
df_valid_review_number = df_valid_date.withColumn( "useful", F.when((F.col("useful").try_cast("int") >= 0), F.col("useful")).otherwise(0) )\
                                      .withColumn( "funny", F.when((F.col("funny").try_cast("int") >= 0), F.col("funny")).otherwise(0) )\
                                      .withColumn( "cool", F.when((F.col("cool").try_cast("int") >= 0), F.col("cool")).otherwise(0) )\
                                      .withColumn( "fans", F.when((F.col("fans").try_cast("int") >= 0), F.col("fans")).otherwise(0) )\
                                      .withColumn( "compliment_hot", F.when((F.col("compliment_hot").try_cast("int") >= 0), F.col("compliment_hot")).otherwise(0) )\
                                      .withColumn( "compliment_more", F.when((F.col("compliment_more").try_cast("int") >= 0), F.col("compliment_more")).otherwise(0) )\
                                      .withColumn( "compliment_profile", F.when((F.col("compliment_profile").try_cast("int") >= 0), F.col("compliment_profile")).otherwise(0) )\
                                      .withColumn( "compliment_list", F.when((F.col("compliment_list").try_cast("int") >= 0), F.col("compliment_list")).otherwise(0) )\
                                      .withColumn( "compliment_note", F.when((F.col("compliment_note").try_cast("int") >= 0), F.col("compliment_note")).otherwise(0) )\
                                      .withColumn( "compliment_plain", F.when((F.col("compliment_plain").try_cast("int") >= 0), F.col("compliment_plain")).otherwise(0) )\
                                      .withColumn( "compliment_cool", F.when((F.col("compliment_cool").try_cast("int") >= 0), F.col("compliment_cool")).otherwise(0) )\
                                      .withColumn( "compliment_funny", F.when((F.col("compliment_funny").try_cast("int") >= 0), F.col("compliment_funny")).otherwise(0) )\
                                      .withColumn( "compliment_writer", F.when((F.col("compliment_writer").try_cast("int") >= 0), F.col("compliment_writer")).otherwise(0) )\
                                      .withColumn( "compliment_photos", F.when((F.col("compliment_photos").try_cast("int") >= 0), F.col("compliment_photos")).otherwise(0) )\
                                      .withColumn( "average_stars", F.when((F.col("average_stars").try_cast("float") >= 0), F.col("average_stars")).otherwise(0) )\
                                      .withColumn( "review_count", F.when((F.col("review_count").try_cast("int") >= 0), F.col("review_count")).otherwise(0) )

# --- Silver: parse and aggregate elite years via UDF ---
def try_cast(val, target_type, default=None):
    """
    Safely cast a value to a target type.

    Used inside UDFs where bad inputs should fall back gracefully without failing the job.
    Accepts any scalar value and type constructor; returns default on cast failure.
    """
    try:
        return target_type(val)
    except (ValueError, TypeError):
        return default
    
def clean_address(text):
    """
    Parse a comma-separated string of elite status years.

    Calculates the boundary years and count of valid elite participation periods.
    Input is a raw delimited string; returns (first_year, last_year, count_years).
    """
    first = "has no elite year"
    last = "has no elite year"
    
    if text is None:
        return first, last, 0
    else:
        list_date = []
        text_split = text.split(",")
        
        for i in text_split:
            if try_cast(i, int):
                # Yelp founded in 2004; elite years before that are invalid data
                if int(i) >= 2004:
                    list_date.append(int(i))
            else:
                # NOTE: purpose unclear
                text_split.remove(i)
                
        list_date.sort()
        if len(list_date) > 0:
            first = str(list_date[0])
            last = str(list_date[-1])
            
    return first, last, len(list_date)

schema = StructType([
    StructField("first_elite_year", StringType(), True),
    StructField("last_elite_year", StringType(), True),
    # count is stored as string to fit the output struct schema definition
    StructField("count_elite_years", StringType(), True)
])            

clean_address_udf = F.udf(clean_address, schema)

df_elite = (
    df_valid_review_number
      .withColumn("new_eilte", clean_address_udf(F.col("elite")))
      .withColumn("first_elite_year", F.col("new_eilte.first_elite_year"))
      .withColumn("last_elite_year", F.col("new_eilte.last_elite_year"))
      .withColumn("user_elite_count", F.col("new_eilte.count_elite_years"))
      .drop("new_eilte", F.col("elite"))
)

# --- Silver: generate surrogate key for dim_user ---
# +1 avoids a 0-value surrogate key to follow standard dimensional modeling practice
df_add_surrogate_key = df_elite.withColumn("User_pk_sk", F.monotonically_increasing_id() + 1)

# --- Silver: project final user dimension schema ---
df_user = df_add_surrogate_key.select(
    F.col("User_pk_sk"),
    
    F.col("User_id"),
    F.col("name").alias("User_name"),   
    
    F.col("first_elite_year"),
    F.col("last_elite_year"),
    F.col("user_elite_count"),
    
    F.col("Join_Date").alias("join_date"),
    F.col("review_count").alias("User_review_count"),
    
    F.col("useful").alias("review_useful"),
    F.col("funny").alias("review_funny"),
    F.col("cool").alias("review_cool"),
    F.col("fans").alias("review_fans"),
    F.col("average_stars").alias("review_average_stars"), 
    
    F.col("compliment_hot").alias("review_compliment_hot"),
    F.col("compliment_more").alias("review_compliment_more"),
    F.col("compliment_profile").alias("review_compliment_profile"),
    F.col("compliment_cute").alias("review_compliment_cute"),
    F.col("compliment_list").alias("review_compliment_list"),
    F.col("compliment_note").alias("review_compliment_note"),
    F.col("compliment_plain").alias("review_compliment_plain"),
    F.col("compliment_cool").alias("review_compliment_cool"),
    F.col("compliment_funny").alias("review_compliment_funny"),
    F.col("compliment_writer").alias("review_compliment_writer"),
    F.col("compliment_photos").alias("review_compliment_photos")
)

# --- Silver: write dim_user to parquet storage ---
df_user.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_user")