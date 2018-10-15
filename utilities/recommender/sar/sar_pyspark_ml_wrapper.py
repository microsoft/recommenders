from pyspark.sql.functions import to_date, unix_timestamp
from pyspark.sql.types import *

data_all = data_all.withColumn('Timestamp', unix_timestamp("date","yyyy-MM-ddTHH:mm:ss:X"))

class RandomSplitter():
    """Random Splitter"""
    def __init__(self, ratio=0.75, seed=123):
      self.ratio=ratio
      if isinstance(ratio, float):
          if ratio <= 0 or ratio >= 1:
              raise ValueError("Split ratio has to be between 0 and 1")
          self.multi_split = False
      elif isinstance(ratio, list):
          if any([x <= 0 for x in ratio]):
              raise ValueError(
                  "All split ratios in the ratio list should be larger than 0.")

          # normalize split ratios if they are not summed to 1
          if sum(ratio) != 1.0:
              ratio = [x / sum(ratio) for x in ratio]

          self.multi_split = True
      else:
          raise TypeError("Split ratio should be either float or a list of floats.")

      self._ratio = ratio
      self.seed = seed

    def split(self, data):
        if self.multi_split:
            return data.randomSplit(self.ratio, seed=self.seed)
        else:
            return data.randomSplit([self.ratio, 1 - self.ratio],
                                    seed=self.seed)
          
header = {
    'col_user': userColIndex,
    'col_item': itemColIndex,
    'col_rating': ratingCol,
    'col_timestamp': "Timestamp"
}


import pyspark.sql.functions as F
from utilities.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    TIMESTAMP_COL,
)

from utilities.recommender.sar import (
    SIM_JACCARD,
    SIM_LIFT,
    SIM_COOCCUR,
    HASHED_USERS,
    HASHED_ITEMS,
)
from utilities.recommender.sar import (
    TIME_DECAY_COEFFICIENT,
    TIME_NOW,
    TIMEDECAY_FORMULA,
    THRESHOLD,
)
from utilities.recommender.sar.sar_pyspark import SARpySparkReference

class SAR():

  def __init__(self, remove_seen=True, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_timestamp=TIMESTAMP_COL, similarity_type=SIM_JACCARD, time_decay_coefficient=TIME_DECAY_COEFFICIENT, time_now=TIME_NOW, timedecay_formula=TIMEDECAY_FORMULA, threshold=THRESHOLD, debug=False):
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    self.sar = SARpySparkReference(spark, remove_seen, col_user, col_item, col_rating, col_timestamp, similarity_type, time_decay_coefficient, time_now, timedecay_formula, threshold, debug)
    self.threshold = threshold
      
  def fit(self, df, epm=None):
    
    # split into two spark dataframes for training and testing
    train, test = RandomSplitter(ratio=0.8).split(df)

    # explicitly make sure we don't have cold users
    train_set_users = set([x[0] for x in train.select(header["col_user"]).distinct().collect()])
    test_set_users = set([x[0] for x in test.select(header["col_user"]).distinct().collect()])
    both_sets = train_set_users.intersection(test_set_users)
    test = test.filter(F.col(header["col_user"]).isin(both_sets))

    """"
    Indexing
    """
    # index the users and items
    train = train.withColumn('type', F.lit(1))
    test = test.withColumn('type', F.lit(0))
    df_all = train.union(test)
    df_all.createOrReplaceTempView("df_all")

    # create new index for the items
    query = "select " + header["col_user"] + ", " +\
        "dense_rank() over(partition by 1 order by " + header["col_user"] + ") as row_id, " +\
                        header["col_item"] + ", " +\
        "dense_rank() over(partition by 1 order by " + header["col_item"] + ") as col_id, " +\
            header["col_rating"] + ", " + header["col_timestamp"] + ", type from df_all"
    log.info("Running query -- " + query)
    df_all = spark.sql(query)
    df_all.createOrReplaceTempView("df_all")

    log.info("Obtaining all users and items ")
    # Obtain all the users and items from both training and test data
    unique_users =\
        np.array([x[header["col_user"]] for x in df_all.select(header["col_user"]).distinct().toLocalIterator()])
    unique_items =\
        np.array([x[header["col_item"]] for x in df_all.select(header["col_item"]).distinct().toLocalIterator()])

    log.info("Indexing users and items")
    # index all rows and columns, then split again intro train and test
    # We perform the reduction on Spark across keys before calling .collect so this is scalable
    index2user = \
        dict(df_all.select(["row_id", header["col_user"]]).rdd.reduceByKey(lambda _, v: v).collect())
    index2item = \
        dict(df_all.select(["col_id", header["col_item"]]).rdd.reduceByKey(lambda _, v: v).collect())

    # reverse the dictionaries: actual IDs to inner index
    user_map_dict = {v: k for k, v in index2user.items()}
    item_map_dict = {v: k for k, v in index2item.items()}

    log.info("Obtain the indexed dataframes")
    query = "select row_id, col_id, " + header["col_rating"] + ", " + header["col_timestamp"] + " from df_all where type=1"
    log.info("Running query -- " + query)
    train_indexed = spark.sql(query)

    query = "select row_id, col_id, " + header["col_rating"] + ", " + header["col_timestamp"] + " from df_all where type=0"
    log.info("Running query -- " + query)
    test_indexed = spark.sql(query)

    # we need to index the train and test sets for SAR matrix operations to work
    # TODO: in MVP3 this index will be passed along with the data structure which we are yet to design.
    self.sar.set_index(unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item)

    self.sar.fit(train_indexed)
    return [SARModel(self.sar, test_indexed)]
        
  def getUserCol(self):
    return self.sar.col_user
  def getRatingCol(self):
    return self.sar.col_rating
  def getItemCol(self):
    return self.sar.col_item
  
class SARModel():
  def __init__(self, sar, df):
    self.sar = sar
    self.df = df
    
  def recommendForAllUsers(self, k):
    recs = self.sar.recommend_k_items(test=self.df, top_k=k)
    from pyspark.sql.functions import col, collect_list
    grouped = recs.groupBy(col('ContactIndex')).agg(collect_list(col("RuleIndex")),collect_list(col("prediction")))

    def Func(lines):
      out = []
      for i in range(len(lines[1])):
        out += [(lines[1][i],lines[2][i])]
      return lines[0], out
    df = grouped.rdd.map(Func).toDF().withColumnRenamed("_1","ContactIndex").withColumnRenamed("_2","recommendations")
    
    tup = StructType([
      StructField(itemColIndex, IntegerType(), True),
      StructField('rating', FloatType(), True)
    ])

    array_type = ArrayType(tup, True)
    return df.select(col("ContactIndex"),col("recommendations").cast(array_type))
    
  def recommendForAllItems(self, k):
    raise NotImplementedError
    
  def transform(self, df):
    return df
