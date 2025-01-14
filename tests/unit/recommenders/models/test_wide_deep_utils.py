# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import pytest
import pandas as pd

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)

try:
    from recommenders.models.wide_deep.wide_deep_utils import (
        WideAndDeep,
        WideAndDeepDataset,
        WideAndDeepHyperParams,
    )
except ImportError:
    pass  # skip this import if we are in cpu environment


ITEM_FEAT_COL = "itemFeat"


@pytest.fixture(scope="module")
def pd_df():
    df = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2],
            DEFAULT_ITEM_COL: [1, 2, 3, 1, 4, 5],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3],
        }
    )
    item_feat = pd.DataFrame({
        DEFAULT_ITEM_COL: [1, 2, 3, 4, 5],
        ITEM_FEAT_COL: [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ],
    })
    users = df.drop_duplicates(DEFAULT_USER_COL)[DEFAULT_USER_COL].values
    items = df.drop_duplicates(DEFAULT_ITEM_COL)[DEFAULT_ITEM_COL].values
    return df, users, items, item_feat

@pytest.mark.gpu
def test_wide_deep_dataset(pd_df):
    data, users, items, item_feat = pd_df
    dataset = WideAndDeepDataset(data)
    assert len(dataset) == len(data)
    # Add +1 because user 0 does count for `dataset`
    assert dataset.n_users == len(users)+1
    assert dataset.n_items == len(items)+1
    assert dataset.n_cont_features == 0
    item, rating = dataset[0]
    assert list(item['interactions']) == [1,1]
    assert 'continuous_features' not in item
    assert rating == 5

    # Test using the item features
    dataset = WideAndDeepDataset(data, item_feat=item_feat)
    assert len(dataset) == len(data)
    # Add +1 because user 0 does count for `dataset`
    assert dataset.n_users == len(users)+1
    assert dataset.n_items == len(items)+1
    assert dataset.n_cont_features == 3
    item, rating = dataset[0]
    assert list(item['interactions']) == [1,1]
    assert list(item['continuous_features']) == [1,1,1]
    assert rating == 5

@pytest.mark.gpu
def test_wide_deep_model(pd_df, tmp):
    data, users, items, item_feat = pd_df

    dataset = WideAndDeepDataset(data)
    default_hparams = WideAndDeepHyperParams()
    model = WideAndDeep(
        dataset,
        dataset,
    )
    
    assert model.model.deep[0].in_features == default_hparams.item_dim + default_hparams.user_dim
    assert model.model.head[-1].out_features == 1

    # Test if the model train works
    model.fit_step()
    assert model.current_epoch == len(model.train_loss_history) == len(model.test_loss_history) == 1

@pytest.mark.gpu
def test_wide_deep_recs(pd_df, tmp):
    data, users, items, item_feat = pd_df

    dataset = WideAndDeepDataset(data)
    model = WideAndDeep(
        dataset,
        dataset,
    )

    recs = model.recommend_k_items(users, items, top_k=4, remove_seen=False)

    assert len(recs) == len(users)*4
    assert set(recs[DEFAULT_USER_COL].unique()) == set(users)
    assert set(recs[DEFAULT_ITEM_COL].unique()).issubset(items)

    # Each user has voted in 3 items, therefore
    # only two items remain to be recommended per user
    # even if we specify top_k>2
    recs = model.recommend_k_items(users, items, top_k=4)
    assert len(recs) == 2*2
    assert set(recs[DEFAULT_USER_COL].unique()).issubset(users)
    assert set(recs[DEFAULT_ITEM_COL].unique()).issubset(items)
