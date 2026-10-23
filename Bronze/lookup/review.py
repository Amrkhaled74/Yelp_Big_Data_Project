#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pathlib import Path

# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")


# In[2]:


# Read the Yelp review dataset from a local JSON file into a Spark DataFrame
# Spark will automatically infer the schema from the JSON structure
df_read_review_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_review.json")


# In[3]:


# Write the DataFrame to Parquet format
df_read_review_data.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/Bronze_Store/review")


# In[ ]:





# In[ ]:




