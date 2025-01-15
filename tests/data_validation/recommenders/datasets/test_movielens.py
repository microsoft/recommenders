# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import gc
import pandas
from pandas.core.series import Series
import pytest
from pytest_mock import MockerFixture

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
    )
    from pyspark.sql.functions import col
except ImportError:
    pass  # skip this import if we are in pure python environment

from recommenders.utils.constants import DEFAULT_GENRE_COL, DEFAULT_TITLE_COL
from recommenders.datasets.movielens import MockMovielensSchema
from recommenders.datasets.movielens import (
    load_pandas_df,
    load_spark_df,
    load_item_df,
    load_user_df,
    download_movielens,
    extract_movielens,
)
from recommenders.datasets.movielens import (
    DATA_FORMAT,
    MOCK_DATA_FORMAT,
    DEFAULT_HEADER,
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
)


@pytest.mark.parametrize("size", [10, 100])
def test_mock_movielens_schema__has_default_col_names(size):
    df = MockMovielensSchema.example(size=size)
    for col_name in DEFAULT_HEADER:
        assert col_name in df.columns


@pytest.mark.parametrize("keep_first_n_cols", [2, 3, 4])
def test_mock_movielens_schema__get_df_remove_default_col__return_success(
    keep_first_n_cols,
):
    df = MockMovielensSchema.get_df(size=3, keep_first_n_cols=keep_first_n_cols)
    assert len(df) > 0
    assert len(df.columns) == keep_first_n_cols


@pytest.mark.parametrize("keep_first_n_cols", [-1, 0, 100])
def test_mock_movielens_schema__get_df_invalid_param__return_failure(keep_first_n_cols):
    with pytest.raises(ValueError, match=r"Invalid value.*"):
        MockMovielensSchema.get_df(size=3, keep_first_n_cols=keep_first_n_cols)


@pytest.mark.parametrize("keep_genre_col", [True, False])
@pytest.mark.parametrize("keep_title_col", [True, False])
@pytest.mark.parametrize("keep_first_n_cols", [None, 2])
@pytest.mark.parametrize("seed", [-1])  # seed for pseudo-random # generation
@pytest.mark.parametrize("size", [0, 3, 10])
def test_mock_movielens_schema__get_df__return_success(
    size, seed, keep_first_n_cols, keep_title_col, keep_genre_col
):
    df = MockMovielensSchema.get_df(
        size=size,
        seed=seed,
        keep_first_n_cols=keep_first_n_cols,
        keep_title_col=keep_title_col,
        keep_genre_col=keep_genre_col,
    )
    assert type(df) == pandas.DataFrame
    assert len(df) == size

    if keep_title_col:
        assert len(df[DEFAULT_TITLE_COL]) == size
    if keep_genre_col:
        assert len(df[DEFAULT_GENRE_COL]) == size


def test_mock_movielens_data__no_name_collision():
    """
    Making sure that no common names are shared between the mock and real dataset sizes
    """
    dataset_name = set(DATA_FORMAT.keys())
    dataset_name_mock = set(MOCK_DATA_FORMAT.keys())
    collision = dataset_name.intersection(dataset_name_mock)
    assert not collision


def test_load_pandas_df_mock_100__with_default_param__succeed():
    df = load_pandas_df("mock100")
    assert type(df) == pandas.DataFrame
    assert len(df) == 100
    assert not df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].duplicated().any()


def test_load_pandas_df_mock_100__with_custom_param__succeed():
    df = load_pandas_df(
        "mock100", title_col=DEFAULT_TITLE_COL, genres_col=DEFAULT_GENRE_COL
    )
    assert type(df[DEFAULT_TITLE_COL]) == Series
    assert type(df[DEFAULT_GENRE_COL]) == Series
    assert len(df) == 100
    assert "|" in df.loc[0, DEFAULT_GENRE_COL]
    assert df.loc[0, DEFAULT_TITLE_COL] == "foo"


@pytest.mark.parametrize("size", ["100k", "1m", "10m", "20m"])
def test_download_and_extract_movielens(size, tmp):
    """Test movielens data download and extract"""
    zip_path = os.path.join(tmp, "ml.zip")
    download_movielens(size, dest_path=zip_path)
    assert len(os.listdir(tmp)) == 1
    assert os.path.exists(zip_path) is True

    rating_path = os.path.join(tmp, "rating.dat")
    item_path = os.path.join(tmp, "item.dat")
    extract_movielens(
        size, rating_path=rating_path, item_path=item_path, zip_path=zip_path
    )
    # Test if raw-zip file, rating file, and item file are cached
    assert len(os.listdir(tmp)) == 3
    assert os.path.exists(rating_path) is True
    assert os.path.exists(item_path) is True


@pytest.mark.parametrize(
    "size, num_samples, num_movies, movie_example, title_example, genres_example, year_example",
    [
        (
            "100k",
            100000,
            1682,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        ),
        (
            "1m",
            1000209,
            3883,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        ),
        (
            "10m",
            10000054,
            10681,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
        (
            "20m",
            20000263,
            27278,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
    ],
)
def test_load_pandas_df(
    size,
    num_samples,
    num_movies,
    movie_example,
    title_example,
    genres_example,
    year_example,
    tmp,
):
    """Test MovieLens dataset load as pd.DataFrame"""
    # Test if correct data are loaded
    header = ["a", "b", "c"]
    df = load_pandas_df(size=size, local_cache_path=tmp, header=header)
    assert len(df) == num_samples
    assert len(df.columns) == len(header)
    # Test if raw-zip file, rating file, and item file are cached
    assert len(os.listdir(tmp)) == 3

    # Test title, genres, and released year load
    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = load_pandas_df(
            size=size,
            header=header,
            local_cache_path=tmp,
            title_col="Title",
            genres_col="Genres",
            year_col="Year",
        )
        assert len(df) == num_samples
        assert (
            len(df.columns) == 7
        )  # 4 header columns (user, item, rating, timestamp) and 3 feature columns
        assert "e" not in df.columns  # only the first 4 header columns are used
        # Get two records of the same items and check if the item-features are the same.
        head = df.loc[df["b"] == movie_example][:2]
        title = head["Title"].values
        assert title[0] == title[1]
        assert title[0] == title_example
        genres = head["Genres"].values
        assert genres[0] == genres[1]
        assert genres[0] == genres_example
        year = head["Year"].values
        assert year[0] == year[1]
        assert year[0] == year_example

    # Test default arguments
    df = load_pandas_df(size)
    assert len(df) == num_samples
    # user, item, rating and timestamp
    assert len(df.columns) == 4
    del df
    gc.collect()


@pytest.mark.parametrize(
    "size, num_movies, movie_example, title_example, genres_example, year_example",
    [
        ("100k", 1682, 1, "Toy Story (1995)", "Animation|Children's|Comedy", "1995"),
        ("1m", 3883, 1, "Toy Story (1995)", "Animation|Children's|Comedy", "1995"),
        (
            "10m",
            10681,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
        (
            "20m",
            27278,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
    ],
)
def test_load_item_df(
    size,
    num_movies,
    movie_example,
    title_example,
    genres_example,
    year_example,
    tmp,
):
    """Test movielens item data load (not rating data)"""
    df = load_item_df(size, local_cache_path=tmp, title_col="title")
    assert len(df) == num_movies
    # movie_col and title_col should be loaded
    assert len(df.columns) == 2
    assert df["title"][0] == title_example

    # Test title and genres
    df = load_item_df(
        size,
        local_cache_path=tmp,
        movie_col="item",
        genres_col="genres",
        year_col="year",
    )
    assert len(df) == num_movies
    # movile_col, genres_col and year_col
    assert len(df.columns) == 3

    assert df["item"][0] == movie_example
    assert df["genres"][0] == genres_example
    assert df["year"][0] == year_example
    del df
    gc.collect()


@pytest.mark.parametrize(
    "size, num_users, user_example, age_example, gender_example, occupation_example, zip_code_example",
    [
        ("100k", 943, 1, 24, "M", "technician", "85711"),
        ("1m", 6040, 1, 1, "F", "K-12 student", "48067"),
    ],
)
def test_load_user_df(
    size,
    num_users,
    user_example,
    age_example,
    gender_example,
    occupation_example,
    zip_code_example,
    tmp,
):
    """Test movielens item data load (not rating data)"""
    df = load_user_df(
        size,
        local_cache_path=tmp,
        user_col="user",
        occupation_col="occupation",
    )
    assert len(df) == num_users
    # user_col and title_col should be loaded
    assert len(df.columns) == 2
    assert df["user"][0] == user_example
    assert df["occupation"][0] == occupation_example

    # Test title and genres
    df = load_user_df(
        size,
        local_cache_path=tmp,
        user_col="user",
        age_col="age",
        gender_col="gender",
        occupation_col="occupation",
        zip_code_col="zip_code",
    )
    assert len(df) == num_users
    # user_col, age_col, gender_col, occupation_col and zip_code_col
    assert len(df.columns) == 5

    assert df["user"][0] == user_example
    assert df["age"][0] == age_example
    assert df["gender"][0] == gender_example
    assert df["occupation"][0] == occupation_example
    assert df["zip_code"][0] == zip_code_example
    del df
    gc.collect()


@pytest.mark.parametrize(
    "size",
    ["10m", "20m"]
)
def test_load_user_df_error(
    size,
    tmp,
):
    assert load_user_df(size, tmp) is None
    assert load_user_df(size, tmp, "user") is None

    with pytest.raises(ValueError, match=".*does not support user info.*"):
        load_user_df(size, tmp, "user", "age")

@pytest.mark.spark
@pytest.mark.parametrize("keep_genre_col", [True, False])
@pytest.mark.parametrize("keep_title_col", [True, False])
@pytest.mark.parametrize("seed", [101])  # seed for pseudo-random # generation
@pytest.mark.parametrize("size", [0, 3, 10])
def test_mock_movielens_schema__get_spark_df__return_success(
    spark, size, seed, keep_title_col, keep_genre_col
):
    df = MockMovielensSchema.get_spark_df(
        spark,
        size=size,
        seed=seed,
        keep_title_col=keep_title_col,
        keep_genre_col=keep_genre_col,
    )
    assert df.count() == size

    if keep_title_col:
        assert df.schema[DEFAULT_TITLE_COL]
    if keep_genre_col:
        assert df.schema[DEFAULT_GENRE_COL]


@pytest.mark.spark
def test_mock_movielens_schema__get_spark_df__store_tmp_file(spark, tmp_path):
    data_size = 3
    MockMovielensSchema.get_spark_df(spark, size=data_size, tmp_path=tmp_path)
    assert os.path.exists(os.path.join(tmp_path, f"mock_movielens_{data_size}.csv"))


@pytest.mark.spark
def test_mock_movielens_schema__get_spark_df__data_serialization_default_param(
    spark, mocker: MockerFixture
):
    data_size = 3
    to_csv_spy = mocker.spy(pandas.DataFrame, "to_csv")

    df = MockMovielensSchema.get_spark_df(spark, size=data_size)
    # assertions
    to_csv_spy.assert_called_once()
    assert df.count() == data_size


@pytest.mark.spark
def test_load_spark_df_mock_100__with_default_param__succeed(spark):
    df = load_spark_df(spark, "mock100")
    assert df.count() == 100


@pytest.mark.spark
def test_load_spark_df_mock_100__with_custom_param__succeed(spark):
    df = load_spark_df(
        spark, "mock100", title_col=DEFAULT_TITLE_COL, genres_col=DEFAULT_GENRE_COL
    )
    assert df.schema[DEFAULT_TITLE_COL]
    assert df.schema[DEFAULT_GENRE_COL]
    assert df.count() == 100
    assert "|" in df.take(1)[0][DEFAULT_GENRE_COL]
    assert df.take(1)[0][DEFAULT_TITLE_COL] == "foo"


@pytest.mark.spark
@pytest.mark.parametrize(
    "size, num_samples, num_movies, movie_example, title_example, genres_example, year_example",
    [
        (
            "100k",
            100000,
            1682,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        ),
        (
            "1m",
            1000209,
            3883,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        ),
        (
            "10m",
            10000054,
            10681,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
        (
            "20m",
            20000263,
            27278,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
    ],
)
def test_load_spark_df(
    size,
    num_samples,
    num_movies,
    movie_example,
    title_example,
    genres_example,
    year_example,
    tmp,
    spark,
):
    """Test MovieLens dataset load into pySpark.DataFrame"""

    # Test if correct data are loaded
    header = ["1", "2", "3"]
    schema = StructType(
        [
            StructField("u", IntegerType()),
            StructField("m", IntegerType()),
        ]
    )
    with pytest.warns(Warning):
        df = load_spark_df(
            spark, size=size, local_cache_path=tmp, header=header, schema=schema
        )
        assert df.count() == num_samples
        # Test if schema is used when both schema and header are provided
        assert len(df.columns) == len(schema)
        # Test if raw-zip file, rating file, and item file are cached
        assert len(os.listdir(tmp)) == 3

    # Test title, genres, and released year load
    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = load_spark_df(
            spark,
            size=size,
            local_cache_path=tmp,
            header=header,
            title_col="Title",
            genres_col="Genres",
            year_col="Year",
        )
        assert df.count() == num_samples
        assert (
            len(df.columns) == 7
        )  # 4 header columns (user, item, rating, timestamp) and 3 feature columns
        assert "e" not in df.columns  # only the first 4 header columns are used
        # Get two records of the same items and check if the item-features are the same.
        head = df.filter(col("b") == movie_example).limit(2)
        title = head.select("Title").collect()
        assert title[0][0] == title[1][0]
        assert title[0][0] == title_example
        genres = head.select("Genres").collect()
        assert genres[0][0] == genres[1][0]
        assert genres[0][0] == genres_example
        year = head.select("Year").collect()
        assert year[0][0] == year[1][0]
        assert year[0][0] == year_example

    # Test default arguments
    df = load_spark_df(spark, size)
    assert df.count() == num_samples
    # user, item, rating and timestamp
    assert len(df.columns) == 4
    del df
    gc.collect()
