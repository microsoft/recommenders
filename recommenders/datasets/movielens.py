# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import re
import random
import shutil
import warnings
import pandas as pd
from typing import Optional
from zipfile import ZipFile
from recommenders.datasets.download_utils import maybe_download, download_path
from recommenders.utils.notebook_utils import is_databricks
from recommenders.utils.constants import (
    DEFAULT_HEADER,
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_GENRE_COL,
)

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        FloatType,
        LongType,
    )
except ImportError:
    pass  # so the environment without spark doesn't break

import pandera as pa
import pandera.extensions as extensions
from pandera import Field
from pandera.typing import Series


class _DataFormat:
    def __init__(
        self,
        sep,
        path,
        has_header=False,
        item_sep=None,
        item_path=None,
        user_path=None,
        item_has_header=False,
    ):
        """MovieLens data format container as a different size of MovieLens data file
        has a different format

        Args:
            sep (str): Rating data delimiter
            path (str): Rating data path within the original zip file
            has_header (bool): Whether the rating data contains a header line or not
            item_sep (str): Item data delimiter
            item_path (str): Item data path within the original zip file
            user_path (str): User data path within the original zip file
            item_has_header (bool): Whether the item data contains a header line or not
        """

        # Rating file
        self._sep = sep
        self._path = path
        self._has_header = has_header

        # Item file
        self._item_sep = item_sep
        self._item_path = item_path
        self._user_path = user_path
        self._item_has_header = item_has_header

    @property
    def separator(self):
        return self._sep

    @property
    def path(self):
        return self._path

    @property
    def has_header(self):
        return self._has_header

    @property
    def item_separator(self):
        return self._item_sep

    @property
    def item_path(self):
        return self._item_path

    @property
    def user_path(self):
        return self._user_path

    @property
    def item_has_header(self):
        return self._item_has_header


# 10m and 20m data do not have user data
DATA_FORMAT = {
    "100k": _DataFormat("\t", "ml-100k/u.data", False, "|", "ml-100k/u.item", "ml-100k/u.user", False),
    "1m": _DataFormat(
        "::", "ml-1m/ratings.dat", False, "::", "ml-1m/movies.dat", "ml-1m/users.dat", False
    ),
    "10m": _DataFormat(
        "::", "ml-10M100K/ratings.dat", False, "::", "ml-10M100K/movies.dat", None, False
    ),
    "20m": _DataFormat(",", "ml-20m/ratings.csv", True, ",", "ml-20m/movies.csv", None, True),
}

# Fake data for testing only
MOCK_DATA_FORMAT = {
    "mock100": {"size": 100, "seed": 6},
}

# 100K data genres index to string mapper. For 1m, 10m, and 20m, the genres labels are already in the dataset.
GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

# 1m data occupation index to string mapper. For 100k, the occupation labels are already in the dataset.
OCCUPATIONS = (
    "Other",
    "Academic/Educator",
    "Artist",
    "Clerical/Admin",
    "College/Grad student",
    "Customer service",
    "Doctor/Health care",
    "Executive/Managerial",
    "Farmer",
    "Homemaker",
    "K-12 student",
    "Lawyer",
    "Programmer",
    "Retired",
    "Sales/Marketing",
    "Scientist",
    "Self-employed",
    "Technician/Engineer",
    "Tradesman/Craftsman",
    "Unemployed",
    "Writer",
)


# Warning and error messages
WARNING_MOVIE_LENS_HEADER = """MovieLens rating dataset has four columns
    (user id, movie id, rating, and timestamp), but more than four column names are provided.
    Will only use the first four column names."""
WARNING_HAVE_SCHEMA_AND_HEADER = """Both schema and header are provided.
    The header argument will be ignored."""
ERROR_MOVIE_LENS_SIZE = (
    "Invalid data size. Should be one of {100k, 1m, 10m, or 20m, or mock100}"
)
ERROR_HEADER = "Header error. At least user and movie column names should be provided"


def load_pandas_df(
    size="100k",
    header=None,
    local_cache_path=None,
    # Movie properties
    title_col=None,
    genres_col=None,
    year_col=None,
    # User properties
    age_col=None,
    gender_col=None,
    occupation_col=None,
    zip_code_col=None,
):
    """Loads the MovieLens dataset as pd.DataFrame.

    Download the dataset from https://files.grouplens.org/datasets/movielens, unzip, and load.
    To load movie information only, you can use load_item_df function.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m", "mock100").
        header (list or tuple or None): Rating dataset header.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored and data is rendered using the 'DEFAULT_HEADER' instead.
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
        age_col (str): User age column name. If None, the column will not be loaded.
        gender_col (str): User gender column name. If None, the column will not be loaded.
        occupation_col (str): User occupation column name. If None, the column will not be loaded.
        zip_code_col (str): User zip code column name. If None, the column will not be loaded.

    Returns:
        pandas.DataFrame: Movie rating dataset.


    **Examples**

    .. code-block:: python

        # To load just user-id, item-id, and ratings from MovieLens-1M dataset,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating'))

        # To load rating's timestamp together,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'))

        # To load movie's title, genres, and released year info along with the ratings data,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col='Title',
            genres_col='Genres',
            year_col='Year'
        )
    """
    size = size.lower()
    if size not in DATA_FORMAT and size not in MOCK_DATA_FORMAT:
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    if header is None:
        header = DEFAULT_HEADER
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]

    if size in MOCK_DATA_FORMAT:
        # generate fake data
        return MockMovielensSchema.get_df(
            keep_first_n_cols=len(header),
            keep_title_col=(title_col is not None),
            keep_genre_col=(genres_col is not None),
            **MOCK_DATA_FORMAT[
                size
            ],  # supply the rest of the kwarg with the dictionary
        )

    user_col = header[0]
    movie_col = header[1]

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        datapath, item_datapath, user_datapath = _maybe_download_and_extract(size, filepath)

        # Load user features such as age, gender, occupation, or zip code
        user_df = _load_user_df(
            size, user_datapath, user_col, age_col, gender_col, occupation_col, zip_code_col
        )

        # Load movie features such as title, genres, and release year
        item_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )

        # Load rating data
        df = pd.read_csv(
            datapath,
            sep=DATA_FORMAT[size].separator,
            engine="python",
            names=header,
            usecols=[*range(len(header))],
            header=0 if DATA_FORMAT[size].has_header else None,
        )

        # Convert 'rating' type to float
        if len(header) > 2:
            df[header[2]] = df[header[2]].astype(float)

        # Merge rating df w/ user_df
        if user_df is not None:
            df = df.merge(user_df, on=header[0])

        # Merge rating df w/ item_df
        if item_df is not None:
            df = df.merge(item_df, on=header[1])

    return df


def load_item_df(
    size="100k",
    local_cache_path=None,
    movie_col=DEFAULT_ITEM_COL,
    title_col=None,
    genres_col=None,
    year_col=None,
):
    """Loads Movie info.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
        movie_col (str): Movie id column name.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.

    Returns:
        pandas.DataFrame: Movie information data, such as title, genres, and release year.
    """
    size = size.lower()
    if size not in DATA_FORMAT:
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        _, item_datapath = _maybe_download_and_extract(size, filepath)
        item_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )

    return item_df


def _load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col):
    """Loads Movie info"""
    if title_col is None and genres_col is None and year_col is None:
        return None

    item_header = [movie_col]
    usecols = [0]

    # Year is parsed from title
    if title_col is not None or year_col is not None:
        item_header.append("title_year")
        usecols.append(1)

    genres_header_100k = None
    if genres_col is not None:
        # 100k data's movie genres are encoded as a binary array (the last 19 fields)
        # For details, see https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
        if size == "100k":
            genres_header_100k = [*(str(i) for i in range(19))]
            item_header.extend(genres_header_100k)
            usecols.extend([*range(5, 24)])  # genres columns
        else:
            item_header.append(genres_col)
            usecols.append(2)  # genres column

    item_df = pd.read_csv(
        item_datapath,
        sep=DATA_FORMAT[size].item_separator,
        engine="python",
        names=item_header,
        usecols=usecols,
        header=0 if DATA_FORMAT[size].item_has_header else None,
        encoding="ISO-8859-1",
    )

    # Convert 100k data's format: '0|0|1|...' to 'Action|Romance|..."
    if genres_header_100k is not None:
        item_df[genres_col] = item_df[genres_header_100k].values.tolist()
        item_df[genres_col] = item_df[genres_col].map(
            lambda l: "|".join([GENRES[i] for i, v in enumerate(l) if v == 1])
        )

        item_df.drop(genres_header_100k, axis=1, inplace=True)

    # Parse year from movie title. Note, MovieLens title format is "title (year)"
    # Note, there are very few records that are missing the year info.
    if year_col is not None:

        def parse_year(t):
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None

        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)

    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)

    return item_df


def load_user_df(
    size="100k",
    local_cache_path=None,
    user_col=DEFAULT_USER_COL,
    age_col=None,
    gender_col=None,
    occupation_col=None,
    zip_code_col=None,
) -> pd.DataFrame:
    """Loads user info

    Args:
        size (str, optional): Size of the data to load. One of ("100k", "1m", "10m", "20m"). Defaults to "100k".
        local_cache_path (str, optional): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be sotred in a temporary directory and removed after use.
        user_col (str): User id column name. Defaults to DEFAULT_USER_COL.
        age_col (str): User age column name. If None, the column will not be loaded.
        gender_col (str): User gender column name (M/F only). If None, the column will not be loaded.
        occupation_col (str): User occupation column name. If None, the column will not be loaded.
        zip_code_col (str): User zip code column name. If None, the column will not be loaded.

    Returns:
        pandas.DatFrame: User information data.
    """
    size = size.lower()

    if size not in DATA_FORMAT:
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        _, _, user_datapath = _maybe_download_and_extract(size, filepath)
        user_df = _load_user_df(
            size, user_datapath, user_col, age_col, gender_col, occupation_col, zip_code_col
        )

    return user_df


def _load_user_df(size, user_datapath, user_col, age_col, gender_col, occupation_col, zip_code_col):
    """Loads user info"""
    if all(c is None for c in [age_col, gender_col, occupation_col, zip_code_col]):
        return None

    if DATA_FORMAT[size].user_path is None:
        raise ValueError(f"Movielens {size} does not support user info. Do not request user info columns.")

    header = {
        0: user_col,
    }

    # 100k has the gender and age columns order swapped
    if age_col is not None:
        if size == '100k':
            header[1] = age_col
        else:
            header[2] = age_col

    if gender_col is not None:
        if size == '100k':
            header[2] = gender_col
        else:
            header[1] = gender_col

    if occupation_col is not None:
        header[3] = occupation_col

    if zip_code_col is not None:
        header[4] = zip_code_col

    usecols = sorted(header.keys())
    user_header = [header[k] for k in usecols]

    user_df = pd.read_csv(
        user_datapath,
        sep=DATA_FORMAT[size].item_separator,
        engine="python",
        names=user_header,
        usecols=usecols,
        header=0 if DATA_FORMAT[size].item_has_header else None,
        encoding="ISO-8859-1",
    )

    # 100k has the labels, but the rest do not
    if size != '100k':
        user_df[occupation_col] = user_df[occupation_col].map(lambda x: OCCUPATIONS[x])

    return user_df


def load_spark_df(
    spark,
    size="100k",
    header=None,
    schema=None,
    local_cache_path=None,
    dbutils=None,
    title_col=None,
    genres_col=None,
    year_col=None,
    age_col=None,
    gender_col=None,
    occupation_col=None,
    zip_code_col=None,
):
    """Loads the MovieLens dataset as `pyspark.sql.DataFrame`.

    Download the dataset from https://files.grouplens.org/datasets/movielens, unzip, and load as `pyspark.sql.DataFrame`.

    To load movie information only, you can use `load_item_df` function.

    Args:
        spark (pyspark.SparkSession): Spark session.
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m", "mock100").
        header (list or tuple): Rating dataset header.
            If `schema` is provided or `size` is set to any of 'MOCK_DATA_FORMAT', this argument is ignored.
        schema (pyspark.StructType): Dataset schema.
            If `size` is set to any of 'MOCK_DATA_FORMAT', data is rendered in the 'MockMovielensSchema' instead.
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
        dbutils (Databricks.dbutils): Databricks utility object
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
        title_col (str): Title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
        age_col (str): User age column name. If None, the column will not be loaded.
        gender_col (str): User gender column name. If None, the column will not be loaded.
        occupation_col (str): User occupation column name. If None, the column will not be loaded.
        zip_code_col (str): User zip code column name. If None, the column will not be loaded.

    Returns:
        pyspark.sql.DataFrame: Movie rating dataset.

    **Examples**

    .. code-block:: python

        # To load just user-id, item-id, and ratings from MovieLens-1M dataset:
        spark_df = load_spark_df(spark, '1m', ('UserId', 'ItemId', 'Rating'))

        # The schema can be defined as well:
        schema = StructType([
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, LongType()),
            ])
        spark_df = load_spark_df(spark, '1m', ('UserId', 'ItemId', 'Rating'), schema=schema)

        # To load rating's timestamp together:
        spark_df = load_spark_df(spark, '1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'))

        # To load movie's title, genres, and released year info along with the ratings data:
        spark_df = load_spark_df(spark, '1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col='Title',
            genres_col='Genres',
            year_col='Year'
        )

        # On DataBricks, pass the dbutils argument as follows:
        spark_df = load_spark_df(spark, dbutils=dbutils)
    """
    size = size.lower()
    if size not in DATA_FORMAT and size not in MOCK_DATA_FORMAT:
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    if size in MOCK_DATA_FORMAT:
        # generate fake data
        return MockMovielensSchema.get_spark_df(
            spark,
            keep_title_col=(title_col is not None),
            keep_genre_col=(genres_col is not None),
            **MOCK_DATA_FORMAT[
                size
            ],  # supply the rest of the kwarg with the dictionary
        )

    schema = _get_schema(header, schema)
    if len(schema) < 2:
        raise ValueError(ERROR_HEADER)

    user_col = schema[0].name
    movie_col = schema[1].name

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        datapath, item_datapath, user_datapath = _maybe_download_and_extract(size, filepath)
        spark_datapath = "file:///" + datapath  # shorten form of file://localhost/

        # Load movie features such as title, genres, and release year.
        # Since the file size is small, we directly load as pd.DataFrame from the driver node
        # and then convert into pyspark.sql.DataFrame
        item_pd_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )
        item_df = spark.createDataFrame(item_pd_df) if item_pd_df is not None else None

        # Load user features such as age, gender, occupation and zip code
        # Since the file size is small, we directly load as pd.DataFrame from the driver node
        # and then convert into pyspark.sql.DataFrame
        user_pd_df = _load_user_df(
            size, user_datapath, user_col, age_col, gender_col, occupation_col, zip_code_col,
        )
        user_df = spark.createDataFrame(user_pd_df) if user_pd_df is not None else None

        if is_databricks():
            if dbutils is None:
                raise ValueError(
                    """
                    To use on a Databricks, dbutils object should be passed as an argument.
                    E.g. load_spark_df(spark, dbutils=dbutils)
                """
                )

            # Move rating file to DBFS in order to load into pyspark.sql.DataFrame
            dbfs_datapath = "dbfs:/tmp/" + datapath
            dbutils.fs.mv(spark_datapath, dbfs_datapath)
            spark_datapath = dbfs_datapath

        # pyspark's read csv currently doesn't support multi-character delimiter, thus we manually handle that
        separator = DATA_FORMAT[size].separator
        if len(separator) > 1:
            raw_data = spark.sparkContext.textFile(spark_datapath)
            data_rdd = raw_data.map(lambda l: l.split(separator)).map(
                lambda c: [int(c[0]), int(c[1]), float(c[2]), int(c[3])][: len(schema)]
            )
            df = spark.createDataFrame(data_rdd, schema)
        else:
            df = spark.read.csv(
                spark_datapath,
                schema=schema,
                sep=separator,
                header=DATA_FORMAT[size].has_header,
            )

        # Merge rating df w/ item_df
        if item_df is not None:
            df = df.join(item_df, movie_col, "left")

        # Merge rating w/ user_df
        if user_df is not None:
            df = df.join(user_df, user_col, "left")

        # Cache and force trigger action since data-file might be removed.
        df.cache()
        df.count()

    return df


def _get_schema(header, schema):
    if schema is None or len(schema) == 0:
        # Use header to generate schema
        if header is None or len(header) == 0:
            header = DEFAULT_HEADER
        elif len(header) > 4:
            warnings.warn(WARNING_MOVIE_LENS_HEADER)
            header = header[:4]

        schema = StructType()
        try:
            (
                schema.add(StructField(header[0], IntegerType()))
                .add(StructField(header[1], IntegerType()))
                .add(StructField(header[2], FloatType()))
                .add(StructField(header[3], LongType()))
            )
        except IndexError:
            pass
    else:
        if header is not None:
            warnings.warn(WARNING_HAVE_SCHEMA_AND_HEADER)

        if len(schema) > 4:
            warnings.warn(WARNING_MOVIE_LENS_HEADER)
            schema = schema[:4]

    return schema


def _maybe_download_and_extract(size, dest_path):
    """Downloads and extracts MovieLens rating and item datafiles if they don’t already exist"""
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    _, rating_filename = os.path.split(DATA_FORMAT[size].path)
    rating_path = os.path.join(dirs, rating_filename)
    _, item_filename = os.path.split(DATA_FORMAT[size].item_path)
    item_path = os.path.join(dirs, item_filename)

    if DATA_FORMAT[size].user_path is None:
        user_path = None
    else:
        _, user_filename = os.path.split(DATA_FORMAT[size].user_path)
        user_path = os.path.join(dirs, user_filename)

    if not all(p is None or os.path.exists(p) for p in [rating_path, item_path, user_path]):
        download_movielens(size, dest_path)
        extract_movielens(size, rating_path, item_path, user_path, dest_path)

    return rating_path, item_path, user_path


def download_movielens(size, dest_path):
    """Downloads MovieLens datafile.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        dest_path (str): File path for the downloaded file
    """
    if size not in DATA_FORMAT:
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    url = "https://files.grouplens.org/datasets/movielens/ml-" + size + ".zip"
    dirs, file = os.path.split(dest_path)
    maybe_download(url, file, work_directory=dirs)


def extract_movielens(size, rating_path, item_path, user_path, zip_path):
    """Extract MovieLens rating and item datafiles from the MovieLens raw zip file.

    To extract all files instead of just rating and item datafiles,
    use ZipFile's extractall(path) instead.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        rating_path (str): Destination path for rating datafile
        item_path (str): Destination path for item datafile
        user_path (str): Destination path for user datafile
        zip_path (str): zipfile path
    """
    with ZipFile(zip_path, "r") as z:
        with z.open(DATA_FORMAT[size].path) as zf, open(rating_path, "wb") as f:
            shutil.copyfileobj(zf, f)
        with z.open(DATA_FORMAT[size].item_path) as zf, open(item_path, "wb") as f:
            shutil.copyfileobj(zf, f)
        if DATA_FORMAT[size].user_path is not None:
            with z.open(DATA_FORMAT[size].user_path) as zf, open(user_path, "wb") as f:
                shutil.copyfileobj(zf, f)


# For more information on data synthesis, see https://pandera.readthedocs.io/en/latest/data_synthesis_strategies.html
@extensions.register_check_method(statistics=["columns"], supported_types=pd.DataFrame)
def unique_columns(df, *, columns):
    return not df[columns].duplicated().any()


class MockMovielensSchema(pa.DataFrameModel):
    """
    Mock dataset schema to generate fake data for testing purpose.
    This schema is configured to mimic the Movielens dataset

    https://files.grouplens.org/datasets/movielens/ml-100k/

    Dataset schema and generation is configured using pandera.
    Please see https://pandera.readthedocs.io/en/latest/schema_models.html
    for more information.
    """

    # Some notebooks will do a cross join with userID and itemID,
    # a sparse range for these IDs can slow down the notebook tests
    userID: Series[int] = Field(
        in_range={"min_value": 1, "max_value": 50}, alias=DEFAULT_USER_COL
    )
    itemID: Series[int] = Field(
        in_range={"min_value": 1, "max_value": 50}, alias=DEFAULT_ITEM_COL
    )
    rating: Series[float] = Field(
        in_range={"min_value": 1, "max_value": 5}, alias=DEFAULT_RATING_COL
    )
    timestamp: Series[int] = Field(
        in_range={"min_value": 0, "max_value": 1e9}, alias=DEFAULT_TIMESTAMP_COL
    )
    title: Series[str] = Field(eq="foo", alias=DEFAULT_TITLE_COL)
    genre: Series[str] = Field(eq="genreA|0", alias=DEFAULT_GENRE_COL)

    @classmethod
    def get_df(
        cls,
        size: int = 3,
        seed: int = 100,
        keep_first_n_cols: Optional[int] = None,
        keep_title_col: bool = False,
        keep_genre_col: bool = False,
    ) -> pd.DataFrame:
        """Return fake movielens dataset as a Pandas Dataframe with specified rows.

        Args:
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.
            keep_first_n_cols (int, optional): keep the first n default movielens columns.
            keep_title_col (bool): remove the title column if False. Defaults to True.
            keep_genre_col (bool): remove the genre column if False. Defaults to True.

        Returns:
            pandas.DataFrame: a mock dataset
        """
        schema = cls.to_schema()
        if keep_first_n_cols is not None:
            if keep_first_n_cols < 1 or keep_first_n_cols > len(DEFAULT_HEADER):
                raise ValueError(
                    f"Invalid value for 'keep_first_n_cols': {keep_first_n_cols}. Valid range: [1-{len(DEFAULT_HEADER)}]"
                )
            schema = schema.remove_columns(DEFAULT_HEADER[keep_first_n_cols:])
        if not keep_title_col:
            schema = schema.remove_columns([DEFAULT_TITLE_COL])
        if not keep_genre_col:
            schema = schema.remove_columns([DEFAULT_GENRE_COL])

        random.seed(seed)
        schema.checks = [pa.Check.unique_columns([DEFAULT_USER_COL, DEFAULT_ITEM_COL])]
        return schema.example(size=size)

    @classmethod
    def get_spark_df(
        cls,
        spark,
        size: int = 3,
        seed: int = 100,
        keep_title_col: bool = False,
        keep_genre_col: bool = False,
        tmp_path: Optional[str] = None,
    ):
        """Return fake movielens dataset as a Spark Dataframe with specified rows

        Args:
            spark (SparkSession): spark session to load the dataframe into
            size (int): number of rows to generate
            seed (int): seeding the pseudo-number generation. Defaults to 100.
            keep_title_col (bool): remove the title column if False. Defaults to False.
            keep_genre_col (bool): remove the genre column if False. Defaults to False.
            tmp_path (str, optional): path to store files for serialization purpose
                when transferring data from python to java.
                If None, a temporal path is used instead

        Returns:
            pyspark.sql.DataFrame: a mock dataset
        """
        pandas_df = cls.get_df(
            size=size, seed=seed, keep_title_col=True, keep_genre_col=True
        )

        # generate temp folder
        with download_path(tmp_path) as tmp_folder:
            filepath = os.path.join(tmp_folder, f"mock_movielens_{size}.csv")
            # serialize the pandas.df as a csv to avoid the expensive java <-> python communication
            pandas_df.to_csv(filepath, header=False, index=False)
            spark_df = spark.read.csv(
                filepath, schema=cls._get_spark_deserialization_schema()
            )
            # Cache and force trigger action since data-file might be removed.
            spark_df.cache()
            spark_df.count()

        if not keep_title_col:
            spark_df = spark_df.drop(DEFAULT_TITLE_COL)
        if not keep_genre_col:
            spark_df = spark_df.drop(DEFAULT_GENRE_COL)
        return spark_df

    @classmethod
    def _get_spark_deserialization_schema(cls):
        return StructType(
            [
                StructField(DEFAULT_USER_COL, IntegerType()),
                StructField(DEFAULT_ITEM_COL, IntegerType()),
                StructField(DEFAULT_RATING_COL, FloatType()),
                StructField(DEFAULT_TIMESTAMP_COL, StringType()),
                StructField(DEFAULT_TITLE_COL, StringType()),
                StructField(DEFAULT_GENRE_COL, StringType()),
            ]
        )
