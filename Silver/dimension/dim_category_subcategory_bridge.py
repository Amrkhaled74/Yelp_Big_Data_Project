#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Set up all required functions
import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType,ArrayType
from itertools import chain
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


# --- Silver: load dimension tables and raw categories source ---
df_dim_subcategory = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_subcategory")

df_dim_category = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_category")

df_read_business_data = spark.read.json("D:/A_Data_engineer/Project_Yelp/Yelp JSON/yelp_academic_dataset_business.json")

df_categories = df_read_business_data.select(F.col("categories"))


# In[ ]:


# --- Silver: extract primary category and subcategories via UDF ---
def get_category_name(text):
    """
    Split category strings into a primary parent category and trailing subcategories.

    Extracts the first comma-separated entry as the parent category and treats remaining elements
    as subcategories, supplying default fallback markers if either segment is missing or empty.
    Input is a raw delimited string; returns a tuple of (primary_category_str, list_of_subcategories).
    """
    if text is None:
        return "Not Specified", ["Not Specified"]

    list_of_element = text.split(",")
    category_name = list_of_element[0].strip()

    # removes the primary parent category from the list to leave only subcategory tokens
    del list_of_element[0]
    if len(list_of_element) == 0:
        return category_name, ["Not Specified"]

    list_of_element_without_space = [st.lower().strip() for st in list_of_element]

    return category_name, list_of_element_without_space

schema = StructType([
    StructField("Categorie_name", StringType(), True),
    StructField("subcategory_names", ArrayType(StringType()), True),
])

get_category_name_udf = F.udf(get_category_name, schema)

# caching avoids re-running the Python UDF for both count materialization and subsequent explode
df_udf = (
    df_categories
    .withColumn("new_eilte", get_category_name_udf(F.col("categories")))
    .withColumn("Categorie_name", F.col("new_eilte.Categorie_name"))
    .withColumn("subcategory_names_list", F.col("new_eilte.subcategory_names"))
    .drop("new_eilte", "categories")
    .cache()
)

# forces completion of the UDF execution before the explode stage begins
df_udf.count()  # Materialization happens at this stage

# --- Silver: explode subcategory arrays into relational rows ---
df_category_subcategory_names = (
    df_udf
    .withColumn("subcategory_names", F.explode(F.col("subcategory_names_list")))
    .drop("subcategory_names_list")
)
df_category_subcategory_names.show(20, truncate=False)


# In[ ]:


# --- Silver: resolve category and subcategory surrogate keys ---
# inner join resolves the primary category string to its Category_pk_sk surrogate key
df_join_get_category_sk = df_category_subcategory_names.join(df_dim_category,on = df_dim_category.Category_name == df_category_subcategory_names.Categorie_name ,how= "inner" )


# In[ ]:


# inner join resolves the subcategory string to its Subcategory_pk_sk surrogate key
df_join_get_subcategory_sk = df_join_get_category_sk.join(df_dim_subcategory,on = df_join_get_category_sk.subcategory_names == df_dim_subcategory.subcategory_name ,how= "inner" )


# In[ ]:


# --- Silver: assemble bridge table schema ---
df_category_subcategory_bridge = df_join_get_subcategory_sk.select(F.col("Category_pk_sk") , F.col("Subcategory_pk_sk") )


# In[ ]:


# --- Silver: write df_category_subcategory_bridge to parquet storage ---

df_category_subcategory_bridge.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_category_subcategory_bridge")


# In[ ]:




