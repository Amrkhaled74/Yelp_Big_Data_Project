#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up all required functions
import os
import sys
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pathlib import Path


# In[2]:


# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")


# In[5]:


# --- Bronze layer: load business source data and target dimensions ---
df_read_business_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_business.json")

df_read_dim_business_data  = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business")


df_hours = df_read_business_data.select(F.col("hours"),F.col("business_id"))


# In[ ]:


# --- Silver: parse operating hours schedule into 7-day normalized arrays ---
def parse_schedule_entry(text):
   """
   Parse raw operating hours strings and standardize into complete 7-day schedules.

   Normalizes variable time strings into 12-hour AM/PM intervals, filling unlisted days with
   sentinel values to preserve aligned array lengths across all days of the week.
   Input is a comma-delimited daily schedule string; returns parallel arrays of (day, start_time, end_time).
   """
   if text is None:
      return ["Not Specified"],["Not Specified"]  ,["Not Specified"]     

   operating_hours = str(text).split(",")
   start  = []
   end = []
   day =[]
   for schedule in operating_hours:
      start_hour="" 
      end_hour="" 
      count = 0

      clean_day_name = schedule.lower().strip()

      if "none" in clean_day_name or  "null" in clean_day_name or (
      "friday" not in clean_day_name and  
      "saturday" not in clean_day_name and 
      "sunday" not in clean_day_name and 
      "monday" not in clean_day_name and
      "tuesday" not in clean_day_name and
      "wednesday" not in clean_day_name and
      "thursday" not in clean_day_name  ) :
         continue

      for slot in schedule:
         if  slot == "-":
            count = 1 

         if slot == ":" and count == 1 :
            end_hour = end_hour + ":"

         if slot == ":" and count == 0 :
            start_hour = start_hour + ":"

         if  slot.isdigit() and count == 1:  
            end_hour=end_hour + slot

         if slot.isdigit() and count == 0:

            start_hour=start_hour + slot

      if not start_hour  or not end_hour :
         continue

      if "friday" in clean_day_name and "friday" not in day:
         day.append("friday")

      if "saturday" in clean_day_name and "saturday" not in day:
         day.append("saturday")

      if "sunday" in clean_day_name and "sunday" not in day:
         day.append("sunday")  

      if "monday" in clean_day_name and "monday" not in day:
         day.append("monday")

      if "tuesday" in clean_day_name and "tuesday" not in day:
         day.append("tuesday")

      if "wednesday" in clean_day_name and "wednesday" not in day:
         day.append("wednesday")

      if "thursday" in clean_day_name and "thursday" not in day:
         day.append("thursday")

      try:
         date_object = datetime.strptime(start_hour, '%H:%M').time()
         start_time_12h = date_object.strftime("%I:%M %p")
      except ValueError:
         start_time_12h = "Time Not Set"

      try:
         date_object = datetime.strptime(end_hour, '%H:%M').time()
         end_time_12h = date_object.strftime("%I:%M %p")
      except ValueError:
         end_time_12h = "Time Not Set"

      # if either boundary fails to parse, treat both bounds as unset to prevent partial interval ranges
      if start_time_12h in  "Time Not Set" or end_time_12h in "Time Not Set" :
        start_time_12h = "Time Not Set" 
        end_time_12h = "Time Not Set"  

      start.append(start_time_12h)
      end.append(end_time_12h )




   # ensure all seven weekdays exist in output arrays to maintain uniform row cardinality per business
   if "friday" not in day:
      day.append("friday")
      start.append("Time Not Set")
      end.append("Time Not Set")

   if "saturday" not in day:
      day.append("saturday")
      start.append("Time Not Set")
      end.append("Time Not Set")

   if "sunday" not in day:
      day.append("sunday")  
      start.append("Time Not Set")
      end.append("Time Not Set")

   if "monday" not in day:
      day.append("monday")
      start.append("Time Not Set")
      end.append("Time Not Set")
   if "tuesday" not in day:
      day.append("tuesday")   
      start.append("Time Not Set")
      end.append("Time Not Set")
   if "wednesday" not in day:
      day.append("wednesday")  
      start.append("Time Not Set")
      end.append("Time Not Set")
   if "thursday" not in day:
      day.append("thursday") 
      start.append("Time Not Set")
      end.append("Time Not Set")
   return day , start , end   


schema = StructType([
    StructField("day_name", ArrayType(StringType()), True), 
    StructField("shift_start_hour", ArrayType(StringType()), True),
    StructField("shift_end_hour", ArrayType(StringType()), True),  
])

parse_schedule_entry_udf = F.udf(parse_schedule_entry, schema)

# cache avoids running the custom Python schedule parser multiple times in subsequent actions
df_elite = (
    df_hours.withColumn("new_eilte", parse_schedule_entry_udf(F.col("hours")))
      .withColumn("day_name", F.col("new_eilte.day_name"))
      .withColumn("shift_start_hour", F.col("new_eilte.shift_start_hour"))
      .withColumn("shift_end_hour", (F.col("new_eilte.shift_end_hour")))
).drop(F.col("new_eilte")).cache()  


# In[ ]:


# --- Silver: unnest parallel day/shift arrays into unique shift combinations ---
# arrays_zip keeps day, start, and end aligned by index before exploding into distinct shift patterns
df_explode = (
    df_elite.withColumn("zipped", F.arrays_zip("day_name", "shift_start_hour", "shift_end_hour"))
      .withColumn("zipped", F.explode(F.col("zipped")))
      .select(
        #  F.col("business_id"),  # or whatever your identifying columns are
          F.col("zipped.day_name").alias("day_name"),
          F.col("zipped.shift_start_hour").alias("shift_start_hour"),
          F.col("zipped.shift_end_hour").alias("shift_end_hour"),
      ).distinct()
)
df_explode.show(truncate=False)


# In[8]:


# --- Silver: initial load surrogate key generation ---
# +1 avoids a 0-value surrogate key to follow standard dimensional modeling practice
df_add_sk= df_explode.withColumn("business_hours_pk_sk", F.monotonically_increasing_id() + 1)  

df_business_hours = df_add_sk.select(F.col("business_hours_pk_sk"),F.col("day_name"),F.col("shift_start_hour"),F.col("shift_end_hour"))


# In[ ]:


# --- Silver: conditional initial seed or incremental append load ---
path = "D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business_hours"

try:
    df_target = spark.read.parquet(path)
    is_empty = df_target.isEmpty()  
except :
    # Triggered if path does not exist or has no valid parquet files
    is_empty = True

if is_empty:
     # initializes dimension store on first pipeline run
     print("")
     #df_business_hours.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business_hours")

else:

    df_read_dim_business_hours = spark.read.parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business_hours")
    # LeftAnti join isolates new schedule shifts that do not already exist in the target dimension
    # Join source and target on ID
    join_condition =  ((df_business_hours["day_name"] == df_read_dim_business_hours["day_name"]  )
                    & (df_business_hours["shift_start_hour"] == df_read_dim_business_hours["shift_start_hour"]) 
                    & (df_business_hours["shift_end_hour"] == df_read_dim_business_hours["shift_end_hour"] ) )




    # identifies schedule shifts present in incoming batch but absent from target storage
    merged_df = df_business_hours.join(df_read_dim_business_hours, join_condition, "LeftAnti")
    # guard prevents executing unnecessary driver actions and writing empty partition files
    if not merged_df.isEmpty ():
        # 1. Coalesce converts NULL max values to 0 on the Spark side
        row = df_read_dim_business_hours.agg(
            F.coalesce(F.max("business_hours_pk_sk"), F.lit(0)).alias("max_business_hours_pk_sk")
        ).first()
        # offsets surrogate keys from existing maximum to maintain continuous unique keys on append
        # 2. Python fallback handles cases where the DataFrame has no rows at all
        mx = row.max_business_hours_pk_sk if (row and row.max_business_hours_pk_sk is not None) else 0
        merged_df_2 = merged_df.withColumn("business_hours_pk_sk", F.monotonically_increasing_id() + F.lit(mx + 1))

        # appends novel operating shift records into the existing parquet dimension store
       # merged_df_2.write.mode("append").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business_hours")


# In[ ]:




