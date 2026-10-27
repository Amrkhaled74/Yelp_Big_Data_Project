#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up all required functions
import os
import sys

from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import  StringType,ArrayType


# In[2]:


# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")


# In[3]:


# --- Silver: load dimension tables and raw categories source ---
df_read_dim_buiness=spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business")

df_read_dim_subcategory = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_subcategory")

df_read_business_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_business.json")

df_read_id_categories= df_read_business_data.select(F.col("business_id"),F.col("categories"))


# In[4]:


# --- Silver: parse subcategory arrays and explode to rows ---
@F.udf(returnType=ArrayType(StringType())) # Define DataFrame schema
def get_subcategory_names(text):
    """
    Extract subcategories from a comma-delimited categories string.

    Treats the first entry as the primary category and removes it to isolate subcategories.
    Input is a raw comma-separated categories string; returns a list of lowercased, trimmed subcategory strings.
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
    df_read_id_categories
    .withColumn("subcategory_name_list", get_subcategory_names(F.col("categories")))
    # NOTE: purpose unclear
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


# --- Silver: project intermediate natural keys for bridge mapping ---
df_business_source =  df_subcategory_name.select(F.col("business_id").alias("business_id_row_source") ,F.col("subcategory_name").alias("subcategory_names") )


# In[6]:


# --- Silver: resolve business surrogate key ---
# match raw business ID to dim_business to fetch Business_pk_sk; filters out unmatched businesses
df_join_to_get_business_id = df_business_source.join(df_read_dim_buiness , on = df_read_dim_buiness.Business_id == df_business_source.business_id_row_source , how="inner") 

df_business = df_join_to_get_business_id.select(F.col("Business_id"),F.col("subcategory_names"),F.col("Business_pk_sk"))


# In[7]:


# --- Silver: resolve subcategory surrogate key ---
# resolve subcategory text name to dim_subcategory surrogate key
df_join_business_subcategory = df_business.join(df_read_dim_subcategory , on = df_business.subcategory_names == df_read_dim_subcategory.subcategory_name ,how= "inner" )


# In[8]:


# --- Silver: build business-to-subcategory bridge table ---
df_business_subcategory_bridge = df_join_business_subcategory.select(F.col("Business_pk_sk"),F.col("Subcategory_pk_sk"))


# In[9]:


# --- Silver: write business_subcategory_bridge to parquet storage ---
#df_business_subcategory_bridge.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/business_subcategory_bridge")


# In[ ]:





# In[ ]:




