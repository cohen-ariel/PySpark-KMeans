# IMPORTS
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import *
from pyspark.sql.window import Window


# AUXILIARY FUNCTIONS
def rename_columns(data, centroids):
    """
    Gets DFs and changes column names. Returns the updated DFs.

    data -> n1, n2, n3 ...
    centroids -> C_n1, C_n2, C_n3 ...
    """
    cnt = 1
    for data_col, cent_col in zip(data, centroids):
        data = data.withColumnRenamed(data_col, f"n{cnt}")
        centroids = centroids.withColumnRenamed(cent_col, f"C_n{cnt}")
        cnt += 1

    return data, centroids


def prepare_data(data_df, centroids_df):
    """
    Rename the 'centroids' DF columns, in order to be different from
    the 'data' DF schema.
    """

    # add indexes (row numbers), 'A' is a dummy value.
    w = Window().orderBy(lit('A'))

    data_df = data_df.withColumn("row_num", row_number().over(w))
    centroids_df = centroids_df.withColumn("c_row_num", row_number().over(w))

    # initial ATTRIBUTE names (not including row numbers)
    temp_cent_attributes = centroids_df.columns[:-1]

    # rename 'centroids' attributes to prepare for cross product
    cent_columns = []
    for att_name in temp_cent_attributes:
        new_att_name = f"C_{att_name}"
        centroids_df = centroids_df.withColumnRenamed(att_name, new_att_name)
        cent_columns.append(new_att_name)
    cent_columns.append("c_row_num")

    return data_df, centroids_df, cent_columns


def cartesian_prod(data_df, centroids_df):
    """Performs cartesian product between the given DFs."""

    # cproduct wil contain BOTH data points AND centroids

    centroids_df = centroids_df.repartition(1000)

    cproduct = data_df.crossJoin(centroids_df)
    return cproduct


def get_distances(cproduct, data_columns, cent_columns):
    """
    Gets DF with data & centroids, and finds the distance between each pair of point and centroid.
    (also column names for both DFs)
    """
    expressions = []
    operation = ""

    for coll in cproduct.columns:
        expressions.append(coll)

    for data_col, cent_col in zip(data_columns, cent_columns):
        operation = operation + f"({data_col} - {cent_col}) * ({data_col} - {cent_col}) + "

    # remove the last ' + '
    operation = operation[:-3]
    expressions.append(operation)

    with_distance = cproduct.selectExpr(expressions)
    with_distance = with_distance.withColumnRenamed(with_distance.columns[-1], 'distance')

    return with_distance


def points_in_which_centroid(with_distance):
    """
    For each point in 'data', finds which centroid it belongs to.
    """

    # for each point find the minimal distance from any centroid.
    w = Window().partitionBy("row_num").orderBy("distance")

    with_min_distance = with_distance.withColumn("min_distance", row_number().over(w)).filter(col("min_distance") == 1) \
        .drop("min_distance")

    return with_min_distance


def get_cluster_avgs(clustered_points, q, cent_columns):
    """
    Gets clustered points and the parameter q, drops out every q'th point and returns
    the attribute-wise average for each cluster. (also drops 'centroids' columns)
    """

    windowSpec = Window.partitionBy("c_row_num").orderBy('distance')

    clustered_points = clustered_points.withColumn("row_num", row_number().over(windowSpec))
    clustered_points.drop('distance')

    clustered_points = clustered_points.filter(col('row_num') % q != 0)

    # drop centroids columns
    for cent_col in cent_columns:
        if cent_col != "c_row_num":
            clustered_points = clustered_points.drop(cent_col)

    # calculate new centroids by using aggregative mean on all relevant columns
    to_remove = ['row_num', 'distance', 'c_row_num']
    to_iter = clustered_points.columns

    for r in to_remove:
        to_iter.remove(r)

    exprs = {x: "avg" for x in to_iter}
    cluster_avgs = clustered_points.groupBy('c_row_num').agg(exprs)

    # clear out the name of attributes from avg()
    # (column names are now in 'avg(...)' form, so change them back to the original names)
    for col_name in cluster_avgs.columns:
        if col_name != 'c_row_num':
            st = col_name[4:-1]
            cluster_avgs = cluster_avgs.withColumnRenamed(col_name, st)

    return cluster_avgs


def restore_centroids_schema(centroids_df):
    """
    Gets DF of centroids and returns it back to the form that we got
    (changes the schema back to normal), And returns the updated DF.
    """
    centroids_df = centroids_df.drop("c_row_num")
    for col_name in centroids_df.columns:
        centroids_df = centroids_df.withColumnRenamed(col_name, col_name[2:])

    return centroids_df


# OUR KMEANS_FIT FUNCTION
def kmeans_fit(data, k, max_iter, q, init):
    """
    Gets data, initial centroids list 'init', num of centroids 'k', drop-out parameter 'q',
    and num of max iterations to do 'max_iter'.

    Performs K-Means algorithm on the data, and returns the centroids it converges to.
    """

    spark = SparkSession.builder.getOrCreate()

    init = spark.createDataFrame(init)

    data_columns = data.columns
    data_df, centroids_df, cent_columns = prepare_data(data, init)

    for i in range(max_iter):
        cproduct = cartesian_prod(data_df, centroids_df)

        with_distance = get_distances(cproduct, data_columns, cent_columns)

        clustered_points = points_in_which_centroid(with_distance)

        new_centroids = get_cluster_avgs(clustered_points, q, cent_columns)

        if len(new_centroids.exceptAll(centroids_df).collect()) == 0:
            break

        # else, keep running
        # change 'centroids' schema back to its original schema
        for col_name in new_centroids.columns:
            if col_name != "c_row_num":
                new_centroids = new_centroids.withColumnRenamed(col_name, f"C_{col_name}")

        centroids_list = new_centroids.collect()
        centroids_df = spark.createDataFrame(centroids_list)

    # update schema before returning
    return restore_centroids_schema(centroids_df)
