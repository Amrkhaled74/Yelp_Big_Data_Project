#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Set up all required functions
import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pathlib import Path


# In[ ]:


# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")


# In[ ]:


# --- Bronze layer: load business source data ---
df_read_business_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_business.json")

df_categories = df_read_business_data.select(F.col("categories"))


# In[ ]:


# --- Silver: extract primary category via UDF ---
@F.udf(returnType=StringType()) # Define DataFrame schema
def get_category_name(text):
    """
    Extract the primary parent category from a comma-separated categories string.

    Treats the first comma-separated token as the primary category name.
    Input is a raw comma-separated string; returns the trimmed primary category name as a string.
    """
    if text is None:
       # NOTE: purpose unclear
       return "Not Specified", ["Not Specified"]

    # first element is defined as the primary category in the Yelp category hierarchy
    list_of_element = text.split(",")
    category_name=list_of_element[0].strip()

    if len(list_of_element) == 0:
        return "Not Specified"

    return category_name


df_Category_name = df_categories.withColumn("Category_name", get_category_name(F.col("categories"))).drop(F.col("categories"))


# In[ ]:


# deduplication ensures each category has exactly one entry before surrogate key allocation
df_distinct_Category_name = df_Category_name.select("Category_name").distinct()


# In[ ]:


# --- Silver: generate surrogate key for dim_category ---
# +1 avoids a 0-value surrogate key to follow standard dimensional modeling practice
df_surrogate_key = df_distinct_Category_name.withColumn("Category_pk_sk", F.monotonically_increasing_id() + 1)


# In[ ]:


# --- Silver: write dim_category to parquet storage ---
df_surrogate_key.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_category")


# In[ ]:




