import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, row_number, max
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.window import *
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, LongType, DoubleType, TimestampType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['CREDENTIALS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['CREDENTIALS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''
    Instantiates and returns a SparkSession configured with the Hadoop-AWS package.
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.1") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Reads in song data from S3, extracts required columns, drops duplicate and null values in ID columns,
        and writes songs and artists tables to S3 as parquet files (partitioned by "year" and "artist_id" 
        columns in the case of the songs table).
        
        Parameters:
            spark: The current SparkSession
            input_data: The path of the source data bucket on S3, formatted with s3a protocol
            output_data: The path of the target bucket for writes on S3, formatted with s3a protocol
    '''
    song_data = input_data + "song_data/*/*/*/*.json"
    
    schema = StructType([
        StructField("num_songs", IntegerType()),
        StructField("artist_id", StringType()),
        StructField("artist_latitude", FloatType()),
        StructField("artist_longitude", FloatType()),
        StructField("artist_location", StringType()),
        StructField("artist_name", StringType()),
        StructField("song_id", StringType()),
        StructField("title", StringType()),
        StructField("duration", FloatType()),
        StructField("year", IntegerType())
    ])
    df = spark.read.json(song_data, schema)

    songs_table = df.select("song_id", "title", "artist_id", "year", "duration") \
        .dropna(subset=["song_id"]).dropDuplicates(subset=["song_id"])
       
    songs_table.write.partitionBy("year", "artist_id").parquet(output_data + "songs.parquet")

    artists_table = df.select("artist_id", 
                              col("artist_name").alias("name"), 
                              col("artist_location").alias("location"), 
                              col("artist_latitude").alias("latitude"), 
                              col("artist_longitude").alias("longitude")) \
                        .dropna(subset=["artist_id"]).dropDuplicates(subset=["artist_id"])
    
    artists_table.write.parquet(output_data + "artists.parquet")


def process_log_data(spark, input_data, output_data):
    '''
    Reads in log data from S3, filters for songplay records, chooses the most recent record associated with a user
        in order to create an up-to-date users table, extracts datetime data to create a timestamp table,
        and joins log and song data to create a songplays table. Duplicate and null values are dropped from users and time
        tables, and each table is written to S3 as a parquet file, partitioned by "year" and "month" columns in the time
        and songplays table.
        
        Parameters:
            spark: The current SparkSession
            input_data: The path of the source data bucket on S3, formatted with s3a protocol
            output_data: The path of the target bucket for writes on S3, formatted with s3a protocol
    '''
    log_data = input_data + "log_data/*/*/*.json"

    schema = StructType([
        StructField("artist", StringType()),
        StructField("auth", StringType()),
        StructField("firstName", StringType()),
        StructField("gender", StringType()),
        StructField("itemInSession", IntegerType()),
        StructField("lastName", StringType()),
        StructField("length", DoubleType()),
        StructField("level", StringType()),
        StructField("location", StringType()),
        StructField("method", StringType()),
        StructField("page", StringType()),
        StructField("registration", DoubleType()),
        StructField("sessionId", IntegerType()),
        StructField("song", StringType()),
        StructField("status", IntegerType()),
        StructField("ts", LongType()),
        StructField("userAgent", StringType()),
        StructField("userId", StringType())
    ])
    df = spark.read.json(log_data, schema)
    
    df = df.where(df.page == "NextSong")

    w = Window.partitionBy("userId")
    users_table = df.dropna(subset=["userId"]) \
                    .withColumn("userid_occurrence_num", row_number() \
                    .over(w.orderBy("ts"))) \
                    .withColumn("max_occurrence_num", max("userid_occurrence_num").over(w)) \
                    .where(col("userid_occurrence_num") == col("max_occurrence_num")) \
                    .select("userId", "firstName", "lastName", "gender", "level")
    
    users_table.write.parquet(output_data + "users.parquet")

    get_timestamp = udf(lambda ts: ts / 1000)
    df = df.withColumn("epoch_ts", get_timestamp(df.ts))
    
    get_datetime = udf(lambda ts: datetime.fromtimestamp(ts), TimestampType())
    df = df.withColumn("datetime", get_datetime(df.epoch_ts))
    
    time_table = df.select("datetime").dropna().dropDuplicates()
    time_table = time_table.withColumn("hour", hour(col("datetime"))) \
        .withColumn("day", dayofmonth(col("datetime"))) \
        .withColumn("week", weekofyear(col("datetime"))) \
        .withColumn("month", month(col("datetime"))) \
        .withColumn("year", year(col("datetime"))) \
        .withColumn("weekday", date_format(col("datetime"), "u"))
    
    time_table.write.partitionBy("year", "month").parquet(output_data + "time.parquet")

    song_df = spark.read.json(input_data + "song_data/*/*/*/*.json",
        StructType([
            StructField("num_songs", IntegerType()),
            StructField("artist_id", StringType()),
            StructField("artist_latitude", FloatType()),
            StructField("artist_longitude", FloatType()),
            StructField("artist_location", StringType()),
            StructField("artist_name", StringType()),
            StructField("song_id", StringType()),
            StructField("title", StringType()),
            StructField("duration", FloatType()),
            StructField("year", IntegerType())
        ])
    )

    df = df.withColumn("songplay_id", row_number() \
                .over(Window.partitionBy("page").orderBy("ts")))

    join_condition = [df.song == song_df.title]
    songplays_table = df.join(song_df, join_condition, "left").select("songplay_id", 
                                                              col("datetime").alias("start_time"),
                                                              col("userId").alias("user_id"),
                                                              "level",
                                                              "song_id",
                                                              "artist_id",
                                                              col("sessionId").alias("session_id"),
                                                              "location",
                                                              col("userAgent").alias("user_agent")) \
                                                            .withColumn("year", year(col("start_time"))) \
                                                            .withColumn("month", month(col("start_time")))

    songplays_table.write.partitionBy("year", "month").parquet(output_data + "songplays.parquet")


def main():
    '''
    Creates a SparkSession, extracts song and log data from S3, transforms it, and loads it back to S3 as parquet files.
    '''
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://another-sample-bucket/output_data/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)
    
    spark.stop()


if __name__ == "__main__":
    main()
