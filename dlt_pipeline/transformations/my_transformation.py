from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import Window

@dp.table(
    name="bronze_taxi",
    table_properties={ "delta.feature.timestampNtz" : "supported" }
)
def bronze_taxi():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .load("/Volumes/databrickstask/raw_files/source_data/taxi/")
    )

@dp.table(
    name="bronze_users",
    table_properties={ "delta.feature.timestampNtz" : "supported" }
)
def bronze_users():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "avro")
        .load("/Volumes/databrickstask/raw_files/source_data/users/")
    )

@dp.table(
    name="bronze_drivers",
    table_properties={ "delta.feature.timestampNtz" : "supported" }
)
def bronze_drivers():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load("/Volumes/databrickstask/raw_files/source_data/drivers/")
    )


@dp.table(
    name="silver_taxi",
    table_properties={ "delta.feature.timestampNtz" : "supported" }
    )
@dp.expect_or_drop("positive_passenger", "passenger_count > 0")
@dp.expect_or_drop("positive_distance", "trip_distance > 0")
@dp.expect_or_drop("positive_fare", "fare_amount > 0")
def silver_taxi():
    return (
        spark
        .readStream
        .table("bronze_taxi")
        .withColumn(
            "description",
             F.when(F.col("payment_type") == 1, "Credit Card")
            .when(F.col("payment_type") == 2, "Cash")
            .otherwise("Other"))
        .dropDuplicates()
        .withColumn("ingest_time", F.current_timestamp()))

@dp.table(
    name="silver_drivers",
    table_properties={ "delta.feature.timestampNtz" : "supported" }
    )
@dp.expect_or_drop("positive_experience", "experience > 0")
@dp.expect_or_drop("valid_rating", "rating BETWEEN 1.0 AND 5.0")
def silver_drivers():
    return (
        spark
        .readStream
        .table("bronze_drivers")
        .dropDuplicates(["name", "car_number"])
        .withColumn("ingest_time", F.current_timestamp())
    )

@dp.table(
    name="silver_users",
    table_properties={ "delta.feature.timestampNtz" : "supported" }
    )
@dp.expect_or_drop("name_not_null", "name IS NOT NULL")
def silver_users():
    return (
        spark
        .readStream
        .table("bronze_users")
        .dropDuplicates()
        .withColumn("ingest_time", F.current_timestamp())
        .withColumn("registration_date", F.current_date())
    )



@dp.materialized_view(
    name="gold_taxi_metrics",
    cluster_by=["PULocationID", "description"]
    )
def gold_taxi_metrics():
    return (
        dp
        .read("silver_taxi")
        .groupBy("PULocationID", "description").agg(
        F.count("*").alias("total_trips"),
        F.round(F.sum("fare_amount"), 2).alias("total_revenue"),
        )
    )   

@dp.materialized_view(
    name="gold_driver_metrics"
    )
def gold_driver_metrics():
    return (
        dp
        .read("silver_drivers")
        .groupBy("experience").agg(
            F.countDistinct("name", "car_number").alias("drivers_count"),
            F.round(F.avg("rating"), 2).alias("avg_rating"),
            )
    )  

@dp.materialized_view(
    name="gold_user_metrics"
    )
def gold_user_metrics():
    return (
        dp
        .read("silver_users")
        .groupBy("registration_date").agg(
            F.count("*").alias("users_count")
            )
        .orderBy(F.desc("users_count"))
    )  

@dp.table(
    name="gold_enriched_trips",
    table_properties={"delta.feature.timestampNtz": "supported"}
)
def gold_enriched_trips():
    taxi = spark.read.table("silver_taxi")
    drivers = spark.read.table("silver_drivers")
    window_spec = Window.orderBy("name", "car_number")
    df_drivers = drivers.withColumn("driver_id", F.row_number().over(window_spec))
    df_n = df_drivers.agg(F.count("*").alias("n"))
    taxi_with_id = (
        taxi
        .crossJoin(df_n)
        .withColumn(
            "driver_id",
            (F.pmod(
                F.hash("PULocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"),
                F.col("n")) + 1).cast("int")
        )
        .drop("n")
    )
    drivers_for_join = df_drivers.select(
        "driver_id",
        F.col("name").alias("driver_name"),
        F.col("car_number").alias("driver_car_number"),
        F.col("experience").alias("driver_experience"),
        F.col("rating").alias("driver_rating"),
    )
    return taxi_with_id.join(drivers_for_join, on="driver_id", how="left")


