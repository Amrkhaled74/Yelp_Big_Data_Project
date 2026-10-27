#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up all required functions
import os
import sys
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.types import StringType,ArrayType


# In[2]:


# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")


# In[3]:


# --- Bronze layer: load business data and reference dimension ---
df_read_business_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_business.json")

df_categories =df_read_business_data.select(F.col("categories"))

df_read_dim_category=spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_category")


# In[4]:


# --- Silver: extract and explode subcategories via UDF ---
@F.udf(returnType=ArrayType(StringType())) # Define DataFrame schema
def get_category_name(text):
    """
    Extract subcategory names from a comma-separated categories string.

    Treats the first entry as the primary category and removes it to isolate subcategories.
    Input is a raw comma-delimited categories string; returns a list of lowercased, trimmed subcategory strings.
    """
    if text is None:
       return ["Not Specified"]

    list_of_element = text.split(",")

    # first element represents the parent category; discarding it isolates subcategories
    del list_of_element[0]

    if len(list_of_element) == 0:
        return  ["Not Specified"]

    list_of_element_without_space = [st.lower().strip() for st in list_of_element]

    return list_of_element_without_space

df_udf = (
    df_categories
    .withColumn("subcategory_name_list", get_category_name(F.col("categories")))
    .drop("new_eilte", "categories")
    # cache avoids re-running the Python UDF for both count materialization and subsequent explode
    .cache()
)

# forces completion of the UDF execution before the explode stage begins
df_udf.count()  # Materialization happens at this stage

df_subcategory_name = (
    df_udf
    .withColumn("subcategory_name", F.explode(F.col("subcategory_name_list")))
    .drop("subcategory_name_list")
)


# In[5]:


# --- Silver: deduplicate and generate surrogate keys for dim_subcategory ---
# deduplication ensures unique subcategory entries prior to surrogate key allocation
df_subcategory_name_distinc = df_subcategory_name.distinct()


# In[6]:


# +1 avoids a 0-value surrogate key to follow standard dimensional modeling practice
df_add_sk= df_subcategory_name_distinc.withColumn("Subcategory_pk_sk", F.monotonically_increasing_id() + 1)


# In[8]:


# --- Silver: write dim_subcategory to parquet storage ---
#df_add_sk.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_subcategory")


# In[ ]:




