import os
import sys
from pyspark.sql import SparkSession
from configuration import settings
from pathlib import Path




def _configure_environment() -> None:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot"
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]

def get_spark_session(app_name: str = "pyspark_ylep", driver_memory: str = "8g") -> SparkSession:
    _configure_environment()

    HADOOP_HOME = Path(os.getenv("HADOOP_HOME", "C:/hadoop")).as_posix()
    SPARK_TMP   = Path(os.getenv("SPARK_TMP", "D:/spark_tmp")).as_posix()

    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.local.dir", SPARK_TMP)
        .config("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={HADOOP_HOME}")
        .getOrCreate()
    )