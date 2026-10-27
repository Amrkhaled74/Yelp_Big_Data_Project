#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Set up all required functions

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.functions import dense_rank, desc, when

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, CharType, ArrayType

from itertools import chain

from pathlib import Path


# In[3]:


# --- Environment setup: configure paths and initialize Spark ---
project_dir = Path(r"D:\A_Data_engineer\Project_Yelp\Bronze\Yelp_Big_Data_Project") 

if str(project_dir) not in sys.path:
    # insert at index 0 to ensure local project modules take precedence over installed packages
    sys.path.insert(0, str(project_dir))

from configuration import get_spark_session

spark = get_spark_session("pyspark_yelp")


# In[4]:


# --- Bronze layer: load raw business dataset ---
df_read_business_data = spark.read.json("D:\\A_Data_engineer\\Project_Yelp\\Yelp JSON\\yelp_academic_dataset_business.json")


# In[5]:


# --- Silver: sanitize address values ---
invalid_placeholders = ["", "N/A", "null", "None", "Confidential"]

# standardizing street abbreviations prevents duplicate location entries in downstream joins
df_modify_address = df_read_business_data.withColumn(
    "new_address",
    F.when(
        F.trim(F.col("address")).isin(invalid_placeholders),
        "unknow address"
    ).otherwise(
        F.regexp_replace(F.lower(F.col("address")), r"\bstreet\b", "st")
    )
)


# In[6]:


# --- Silver: standardize city names and strip administrative noise ---
replacements = {
    r"-davidson metropolitan government \(balance\)": "",    
    r"\s*city \(balance\)": "",    
    r"\s*\(meridian\)": "", 
    r"\s*\(walsingham\)": "",
    r"\s*\(northeast philadelphia\)": "",
    r"\bfl\b\s*": "",
    r"\s*/.*$": "",
    r"\bn\b\.\s*": "north ",
    r"\bst\b\.\s*": "saint ",
    # expands specific known abbreviated municipality from source extract
    r"\bmt\b\.\s*": "mount juliet ",
    r"\bhts\b\.\s*": "heights ",
    r"\be\b\.\s*": "east ",
    r"\bw\b\.\s*": "west ",
    r"\bkop\b\s*": "king of prussia ", 
    r"\btwp\b\s*": "township ", 
    r"\bnj\b\s*": "new jersey ", 
    r"\bphilly\b\s*": "philadelphia ",
    ",": "",
    "&": "and",
}

expr = F.lower(F.col("city"))

for old, new in replacements.items(): 
    expr = F.regexp_replace(expr, old, new)

df_city = df_modify_address.withColumn("cleaned_city", F.trim(expr))


# In[7]:


# --- Silver: validate state values against postal code leakages ---
combined_regex = r"^\d{5}(-\d{4})?$|^(?i)[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ ]?\d[ABCEGHJ-NPRSTV-Z]\d$"

# flags misaligned source fields where postal codes were parsed into the state attribute
df_state = df_city.withColumn(
    "new_state",
    when(F.col("state").isNull(), "unknown state")
    .when(F.upper(F.col("state")).rlike(combined_regex), "unknown state")
    .otherwise(F.upper(F.col("state")))
)


# In[8]:


# --- Silver: load geographic reference lookup table ---
df_state_teritory = spark.read \
    .option("header", "true") \
    .option("sep", ",") \
    .csv("D:\\A_Data_engineer\\Project_Yelp\\US_GeoCode.csv")


# In[9]:


# --- Silver: impute missing or out-of-bounds latitude coordinates ---
dict_state = dict()

for i, j in df_state_teritory.select("state&teritory", "latitude").collect():
    dict_state[i] = j

col_latitude = F.col("latitude").cast("float")

valid_condition = col_latitude.isNotNull() & col_latitude.between(-90.0, 90.0) 

# create_map inlines the small lookup table to avoid a full shuffle join across partitions
map_expr = F.create_map([F.lit(x) for x in chain(*dict_state.items())])

df_latitude = df_state.withColumn(
    "new_latitude",
    when(valid_condition, col_latitude).otherwise(map_expr.getItem(F.col("state")))
)


# In[10]:


# --- Silver: impute missing or out-of-bounds longitude coordinates ---
dict_longitude = dict()

for i, j in df_state_teritory.select("state&teritory", "longitude").collect():
    dict_longitude[i] = j

col_longitude = F.col("longitude").cast("float")

valid_condition = col_longitude.isNotNull() & col_longitude.between(-180.0, 180.0) 

# inlined map expression replaces shuffle join overhead for driver-collected coordinates
map_expr = F.create_map([F.lit(x) for x in chain(*dict_longitude.items())])

df_longitude = df_latitude.withColumn(
    "new_longitude",
    when(valid_condition, col_longitude).otherwise(map_expr.getItem(F.col("state")))
)


# In[11]:


# --- Silver: validate ratings, review counts, and operational flags ---
stars = F.col("stars").try_cast("float")
review_count = F.col("review_count").try_cast("int") 
is_open = F.col("is_open").try_cast("int")

# bounds checks ensure business KPIs remain within measurable domain constraints
df_valid_review_number = df_longitude.withColumn("new_stars", F.when((stars >= 0) & (stars <= 5), stars).otherwise(0))\
                                     .withColumn("new_review_count", F.when((review_count >= 0), review_count).otherwise(0))\
                                     .withColumn("new_is_open", F.when(is_open == 1, 1).otherwise(0))


# In[ ]:


# --- Silver: parse semi-structured attributes string via UDF ---
def Business_attribute(text):
    """
    Parse a comma-separated key-value attributes string into standardized business features.

    Normalizes inconsistent free-text Yelp tags into uniform categorical values for analytics.
    Input is a raw comma-delimited attributes string; returns a tuple of 12 normalized string attributes.
    """
    wifi = ""
    alcohol = ""
    RestaurantsDelivery = ""
    RestaurantsTakeOut = ""
    NoiseLevel = ""
    GoodForKids = ""
    RestaurantsAttire = ""
    RestaurantsReservations = ""
    OutdoorSeating = ""
    Open24Hours = ""
    RestaurantsPriceRange = ""
    BusinessParking = ""

    set_attributes = set()
    split_string = str(text).split(",")
    for values_of_split_string in split_string:
        set_attributes.add(values_of_split_string)

    for individual_attribute_ in set_attributes:
        attribute = individual_attribute_.lower()

        if ("wifi") in attribute:
            if ("paid") in attribute:
                wifi = "Paid"
            elif ("free") in attribute:
                wifi = "Free"
            else:
                wifi = "Not Available"

        if ("alcohol") in attribute:
            if ("beer") in attribute:
                alcohol = "Beer And Wine"
            elif ("bar") in attribute:
                alcohol = "Full Bar"
            else:
                alcohol = "Not Available"

        if ("delivery") in attribute:
            if ("true") in attribute:
                RestaurantsDelivery = "Available"
            else:
                RestaurantsDelivery = "Not Available"  

        if ("takeout") in attribute:
            if ("true") in attribute:
                RestaurantsTakeOut = "Available"
            else:
                RestaurantsTakeOut = "Not Available"  

        if ("noise") in attribute:
            if ("loud") in attribute:
                NoiseLevel = "loud"
            elif ("average") in attribute:
                NoiseLevel = "average"
            elif ("very_loud") in attribute:
                NoiseLevel = "very_loud"
            elif ("quiet") in attribute:
                NoiseLevel = "quiet"
            else:
                NoiseLevel = "Not Specified"                    

        if ("goodforkids") in attribute:
            if ("true") in attribute:
                GoodForKids = "Suitable"
            else:
                GoodForKids = "Not Suitable"                  

        if ("attire") in attribute:
            if ("dressy") in attribute:
                RestaurantsAttire = "dressy"
            elif ("formal") in attribute:
                RestaurantsAttire = "formal"
            else:
                RestaurantsAttire = "casual"                  

        if ("reservations") in attribute:
            if ("true") in attribute:
                RestaurantsReservations = "accepted"
            else:
                RestaurantsReservations = "Not accepted"     

        if ("outdoorseating") in attribute:
            if ("true") in attribute:
                OutdoorSeating = "Available"
            else:
                OutdoorSeating = "Not Available"   

        if ("open24hours") in attribute:
            if ("true") in attribute:
                Open24Hours = "Open 24 Hours"
            else:
                Open24Hours = "Standard Hours"  

        if ("pricerange") in attribute:
            if ("1") in attribute:
                RestaurantsPriceRange = "Inexpensive"
            elif ("2") in attribute:
                RestaurantsPriceRange = "Moderate"
            elif ("3") in attribute:
                RestaurantsPriceRange = "Pricey"
            elif ("4") in attribute:
                RestaurantsPriceRange = "Ultra"
            else:
                RestaurantsPriceRange = "Not Specified" 

        if ("businessparking") in attribute:
            if ("garage") in attribute:
                if ("true") in attribute:
                    BusinessParking = "Available"
            else:
                BusinessParking = "Not Available"                  

    # defaults prevent null propagation across dimensional analytical cubes
    if not wifi:
        wifi = "Not Available"   

    if not alcohol:
        alcohol = "Not Available"   

    if not RestaurantsDelivery:
        RestaurantsDelivery = "Not Available" 

    if not RestaurantsTakeOut:
        RestaurantsTakeOut = "Not Available"    

    if not NoiseLevel:
        NoiseLevel = "Not Specified"     

    if not GoodForKids:
        GoodForKids = "Not Suitable"   

    if not RestaurantsAttire:
        RestaurantsAttire = "casual" 

    if not RestaurantsReservations:
        RestaurantsReservations = "Not accepted" 

    if not OutdoorSeating:
        OutdoorSeating = "Not Available"   

    if not Open24Hours:
        Open24Hours = "Standard Hours"  

    if not RestaurantsPriceRange:
        RestaurantsPriceRange = "Not Specified" 

    if not BusinessParking:
        BusinessParking = "Not Available" 

    return (
        wifi, alcohol, RestaurantsDelivery, RestaurantsTakeOut, NoiseLevel,
        GoodForKids, RestaurantsAttire, RestaurantsReservations, OutdoorSeating,
        BusinessParking, Open24Hours, RestaurantsPriceRange
    )

schema = StructType([
    StructField("WiFi", StringType(), True),
    StructField("Alcohol", StringType(), True),
    StructField("BusinessDelivery", StringType(), True),
    StructField("RestaurantsTakeOut", StringType(), True),
    StructField("NoiseLevel", StringType(), True),
    StructField("GoodForKids", StringType(), True),
    StructField("RestaurantsAttire", StringType(), True),
    StructField("RestaurantsReservations", StringType(), True),
    StructField("OutdoorSeating", StringType(), True),
    StructField("BusinessParking", StringType(), True),
    StructField("Open24Hours", StringType(), True),
    StructField("RestaurantsPriceRange", StringType(), True)
])            

Business_attribute_udf = F.udf(Business_attribute, schema)

# unpacking struct columns into top-level fields flattens attributes for dimensional storage
df_Business_Attribute = (
    df_valid_review_number.withColumn("new_eilte", Business_attribute_udf(F.col("attributes")))
      .withColumn("new_wifi", F.col("new_eilte.WiFi"))
      .withColumn("new_alcohol", F.col("new_eilte.Alcohol"))
      .withColumn("new_BusinessDelivery", F.col("new_eilte.BusinessDelivery"))
      .withColumn("new_RestaurantsTakeOut", F.col("new_eilte.RestaurantsTakeOut"))
      .withColumn("new_NoiseLevel", F.col("new_eilte.NoiseLevel"))
      .withColumn("new_GoodForKids", F.col("new_eilte.GoodForKids"))
      .withColumn("new_RestaurantsAttire", F.col("new_eilte.RestaurantsAttire"))
      .withColumn("new_RestaurantsReservations", F.col("new_eilte.RestaurantsReservations"))
      .withColumn("new_OutdoorSeating", F.col("new_eilte.OutdoorSeating"))
      .withColumn("new_BusinessParking", F.col("new_eilte.BusinessParking"))
      .withColumn("new_Open24Hours", F.col("new_eilte.Open24Hours"))
      .withColumn("new_RestaurantsPriceRange", F.col("new_eilte.RestaurantsPriceRange"))      
      .drop("new_eilte")
)


# In[ ]:


# --- Silver: generate surrogate key for dim_business ---
# +1 avoids a 0-value surrogate key to follow standard dimensional modeling practice
df_Business_surrogate_key = df_Business_Attribute.withColumn("Business_pk_sk", F.monotonically_increasing_id() + 1)


# In[ ]:


# --- Silver: project final business dimension schema ---
df_Business = df_Business_surrogate_key.select(
    F.col("Business_pk_sk"),
    
    F.col("business_id").alias("Business_id"),
    F.col("name").alias("Business_name"),   
    F.col("cleaned_city").alias("city"),
    F.col("new_state").alias("State"),
    F.col("postal_code"),
    
    F.col("new_stars").alias("Business_stars"),
    F.col("new_review_count").alias("Business_review_count"),
    
    F.col("new_RestaurantsPriceRange").alias("Business_price_range"),
    F.col("new_wifi").alias("Business_wifi"),
    F.col("new_alcohol").alias("Business_alcohol"),
    F.col("new_BusinessDelivery").alias("Business_delivery"),
    F.col("new_RestaurantsTakeOut").alias("Business_TakeOut"), 
    F.col("new_NoiseLevel").alias("Business_noiselevel"),
    F.col("new_GoodForKids").alias("Business_goodforkids"),
    F.col("new_RestaurantsAttire").alias("Business_attire"),
    F.col("new_RestaurantsReservations").alias("Business_reservations"),
    F.col("new_OutdoorSeating").alias("Business_seating"),
    F.col("new_BusinessParking").alias("Business_parking"),
    F.col("new_Open24Hours").alias("Business_open24hours"),
    
    F.col("new_is_open").alias("Business_is_open"),
)




# In[ ]:


# --- Silver: write dim_business to parquet storage ---
#df_Business.write.mode("overwrite").parquet("D:/A_Data_engineer/Project_Yelp/store_dimesnion_user/dim_business")


# In[ ]:




