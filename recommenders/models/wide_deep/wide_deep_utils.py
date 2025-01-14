# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
from typing import Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn

from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_PREDICTION_COL
import recommenders.utils.python_utils as pu
import recommenders.utils.torch_utils as tu

@dataclass(frozen=True)
class WideAndDeepHyperParams:
    user_dim: int = 32
    item_dim: int = 32
    crossed_feat_dim: int = 1000
    dnn_hidden_units: Tuple[int, ...] = (128, 128)
    dnn_dropout: float = 0.0
    dnn_additional_embeddings_sizes: dict[str, Tuple[int, int]] = field(default_factory=dict)
    dnn_cont_features: int = 0

class WideAndDeepModel(nn.Module):
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        hparams: WideAndDeepHyperParams = WideAndDeepHyperParams(),
        binary_output: bool = False,
    ):
        super().__init__()

        self.hparams = hparams
        self.n_users = num_users
        self.n_items = num_items
        
        self.users_emb = nn.Embedding(num_users, hparams.user_dim)
        self.items_emb = nn.Embedding(num_items, hparams.item_dim)
        self.additional_embs = nn.ModuleDict({
            k: nn.Embedding(num, dim) for k, (num, dim) in hparams.dnn_additional_embeddings_sizes.items()
        })

        # Randomly initialize embeddings
        total_emb_dim = hparams.user_dim + hparams.item_dim
        for _, emb in self.additional_embs.items():
            total_emb_dim += emb.embedding_dim
            nn.init.uniform_(emb.weight, -1, 1)

        layers = []
        prev_output = hparams.dnn_cont_features + total_emb_dim
        for hu in hparams.dnn_hidden_units:
            layers.append(nn.Linear(prev_output, hu))
            layers.append(nn.Dropout(hparams.dnn_dropout))
            layers.append(nn.ReLU())
            prev_output = hu

        self.deep = nn.Sequential(*layers)

        # Cross product of users-items
        exclusive_wide_input = hparams.crossed_feat_dim

        self.head = nn.Sequential(
            nn.Linear(exclusive_wide_input+prev_output, 1),
        )

        if binary_output:
            self.head.append(nn.Sigmoid())

    def forward(
        self, 
        interactions: torch.Tensor, 
        additional_embeddings: Dict[str, torch.Tensor] = {},
        continuous_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        users, items = interactions.T
    
        all_embed = torch.cat([
            self.users_emb(users), # Receives the indices
            self.items_emb(items),
            *[ emb(additional_embeddings[k]) for k, emb in self.additional_embs.items() ]
        ], dim=1)

        # TODO: Use hashing to avoid problems with biased distributions
        cross_product_idx = (users*self.n_items + items) % self.hparams.crossed_feat_dim
        cross_product = nn.functional.one_hot(cross_product_idx, self.hparams.crossed_feat_dim)

        if self.hparams.dnn_cont_features > 0:
            deep_input = torch.cat([continuous_features, all_embed], dim=1)
        else:
            deep_input = all_embed

        return self.head(torch.cat([
            cross_product, # wide input
            self.deep(deep_input), # deep output
        ], dim=1))


class WideAndDeepDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        ratings: pd.DataFrame, 
        user_col: str = DEFAULT_USER_COL,
        item_col: str = DEFAULT_ITEM_COL,
        rating_col: str = DEFAULT_RATING_COL,
        n_users: Optional[int] = None, 
        n_items: Optional[int] = None,
        item_feat: Optional[pd.DataFrame] = None,
        user_feat: Optional[pd.DataFrame] = None,
        n_cont_features: Optional[int] = None,
    ):
        self._check_cols_df('ratings', ratings, [user_col, item_col, rating_col])
        self._check_cols_df('item_feat', item_feat, [item_col])
        self._check_cols_df('user_feat', user_feat, [user_col])
        self._check_cont_features('item_feat', item_feat, item_col)
        self._check_cont_features('user_feat', user_feat, user_col)

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.ratings = ratings.copy()
        self.item_feat = item_feat.set_index(item_col).copy() if item_feat is not None else pd.DataFrame()
        self.user_feat = user_feat.set_index(user_col).copy() if user_feat is not None else pd.DataFrame()
        self.n_cont_features = n_cont_features or self._get_continuous_features([self.item_feat.index.min()], [self.user_feat.index.min()]).shape[1]

        self.n_users = n_users or ratings[user_col].max()+1
        self.n_items = n_items or ratings[item_col].max()+1

        self.ratings[rating_col] = self.ratings[rating_col].astype('float32')

    @staticmethod
    def _check_cols_df(df_name: str, df: Optional[pd.DataFrame], cols: list[str]) -> bool:
        if df is None or df.empty:
            return True
        
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' is not present on {df_name}")

        return True

    @staticmethod
    def _check_cont_features(df_name: str, df: Optional[pd.DataFrame], col: str) -> bool:
        if df is None or df.empty:
            return True

        for c in df.columns:
            if c == col:
                continue
            
            # Check that dtype is float
            if not pd.api.types.is_numeric_dtype(df[c].dtype):
                raise ValueError(f"Column '{c}' from {df_name} has dtype {df[c].dtype}. Only numeric dtypes are allowed.")
    
    def __len__(self):
        return len(self.ratings)

    def _get_continuous_features(self, item_ids, user_ids) -> np.array:
        if len(item_ids) != len(user_ids):
            raise ValueError('item_ids and user_ids should have same length')

        if self.item_feat.empty and self.user_feat.empty:
            return np.array([])
        
        # Put empty array so concat doesn't fail
        continuous_features = []

        if not self.item_feat.empty:
            feats = self.item_feat.loc[item_ids]
            continuous_features.extend(feats.values)

        if not self.user_feat.empty:
            feats = self.user_feat.loc[user_ids]
            continuous_features.extend(feats.values)

        return np.stack(continuous_features)

    def __getitem__(self, idx):
        # TODO: Get additional embeddings too (e.g: user demographics)
        item = self.ratings.iloc[idx]

        ret = { 
            'interactions': self.ratings[[self.user_col, self.item_col]].iloc[idx].values,
        }

        if self.n_cont_features:
            # Reshape because it is only one item
            ret['continuous_features'] = self._get_continuous_features([item[self.item_col]], [item[self.user_col]]).reshape(-1)

        return ret, self.ratings[self.rating_col].iloc[idx]
    

class WideAndDeep(object):
    def __init__(
        self, 
        train: WideAndDeepDataset,
        test: WideAndDeepDataset,
        hparams: WideAndDeepHyperParams = WideAndDeepHyperParams(),
        *,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        epochs: int = 100,
        batch_size: int = 128,
        loss_fn: Union[str, nn.Module] = 'mse',
        optimizer: str = 'sgd',
        l1: float = 0.0001,
        optimizer_params: dict[str, Any] = dict(),
        disable_batch_progress: bool = False,
        disable_iter_progress: bool = False,
        eval_epoch: int = 1,
        model_dir: Optional[Union[str, Path]] = None,
        save_model_iter: int = -1,
        prediction_col: str = DEFAULT_PREDICTION_COL,
    ):
        self.n_users = n_users or max(train.n_users, test.n_users)
        self.n_items = n_items or max(train.n_items, test.n_items)

        if train.n_cont_features != test.n_cont_features:
            raise ValueError(f'The number of cont. features on the train dataset is not the same as in test')
        if train.n_cont_features != hparams.dnn_cont_features:
            raise ValueError(
                f"The number of cont. features on the dataset ({train.n_cont_features}) "
                f"is not the same as in the hparams ({hparams.dnn_cont_features})"
            )

        self.train = train
        self.test = test
        self.train_dataloader = DataLoader(train, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test, len(test))
        
        self.model = WideAndDeepModel(
            num_users=self.n_users,
            num_items=self.n_items,
            hparams=hparams,
        )
        
        if isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            self.loss_fn = tu.LOSS_DICT[loss_fn]()
        
        self.optimizer = tu.OPTIM_DICT[optimizer](
            self.model.parameters(), 
            lr=l1,
            **optimizer_params,
        )

        self.disable_batch_progress = disable_batch_progress
        self.disable_iter_progress = disable_iter_progress
        self.prediction_col = prediction_col

        self.current_epoch = 0
        self.epochs = epochs
        self.eval_epoch = eval_epoch

        self.model_dir = Path(model_dir) if model_dir else None
        self.save_model_iter = save_model_iter
        self._check_save_model()

        self.train_loss_history = list()
        self.test_loss_history = list()

    @property
    def user_col(self) -> str:
        return self.train.user_col

    @property
    def model_path(self) -> Path:
        return self.model_dir / f'wide_deep_state_{self.current_epoch:05d}.pth'

    @property
    def item_col(self) -> str:
        return self.train.item_col

    def _check_save_model(self) -> bool:
        # The two conditions should be True/False at the same time
        if (self.save_model_iter == -1) != (self.model_dir is None):
            raise ValueError('You should set both save_model_iter and model_dir at the same time')

        if self.model_dir is not None:
            # Check that save works
            self.save()
        
        return True
        
    def fit(self):
        if self.current_epoch >= self.epochs:
            print(f"Model is already trained with {self.epochs} epochs. Increment the number of epochs.")
        
        with tqdm(total=self.epochs, leave=True, disable=self.disable_iter_progress) as pbar:
            pbar.update(self.current_epoch)
            for _ in range(self.current_epoch, self.epochs):
                self.fit_step()
                pbar.update()
                pbar.set_postfix(
                    train_loss=self.train_loss_history[-1],
                    test_loss=self.test_loss_history[-1][1],
                )

                if self.save_model_iter != -1 and self.current_epoch % self.save_model_iter == 0:
                    self.save()

    def save(self, model_path=None):
        model_path = Path(model_path) if model_path else self.model_path
        model_path.parent.mkdir(exist_ok=True)
        
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path=None):
        if model_path is None:
            print('Model path not specified, automatically loading from model dir')
            model_path = max(self.model_dir.glob('*.pth'), key=lambda f: int(f.stem.split('_')[-1]))
            print('  Loading', model_path)
        else:
            model_path = Path(model_path)
            
        self.model.load_state_dict(torch.load(model_path))
        self.current_epoch = int(model_path.stem.split('_')[-1])

    def fit_step(self):
        self.model.train()
        
        train_loss = 0.0
        for X,y in tqdm(self.train_dataloader, 'batch', leave=False, disable=self.disable_batch_progress):
            pred = self.model(
                X['interactions'],
                continuous_features=X.get('continuous_features', None),
            )
            loss = self.loss_fn(pred, y)
            # TODO: Can we use this loss? Or should I calculate it again with no_grad?
            train_loss += loss.item()

            # Propagate error
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.train_loss_history.append(train_loss / len(self.train_dataloader))
        self.model.eval()
        
        if self.eval_epoch != -1 and self.current_epoch%self.eval_epoch == 0:
            num_batches = len(self.test_dataloader)
            test_loss = 0

            with torch.no_grad():
                for X, y in self.test_dataloader:
                    pred = self.model(
                        X['interactions'],
                        continuous_features=X.get('continuous_features', None),
                    )
                    test_loss += self.loss_fn(pred, y).item()

            test_loss /= num_batches
            self.test_loss_history.append((self.current_epoch, test_loss))
    
        self.current_epoch += 1

    def _get_uip_cont(self, user_ids, item_ids, remove_seen: bool):
        if user_ids is None:
            user_ids = np.arange(1, self.n_users)
        if item_ids is None:
            item_ids = np.arange(1, self.n_items)

        uip = pd.MultiIndex.from_product(
            [user_ids, item_ids], 
            names=[self.user_col, self.item_col],
        )

        if remove_seen:
            uip = uip.difference(
                self.train.ratings.set_index([self.user_col, self.item_col]).index
            )

        uip = uip.to_frame(index=False)

        cont_features = None
        if self.train.n_cont_features > 0:
            cont_features = torch.from_numpy(
                self.train._get_continuous_features(uip.values[:,0], uip.values[:,1])
            )

        return uip, cont_features

    def recommend_k_items(
        self, user_ids=None, item_ids=None, top_k=10, remove_seen=True,
    ):
        uip, cont_features = self._get_uip_cont(user_ids, item_ids, remove_seen)
        
        with torch.no_grad():
            uip[self.prediction_col] = self.model(
                torch.from_numpy(uip[[self.user_col, self.item_col]].values),
                continuous_features=cont_features,
            )

        return (
            uip
            .sort_values([self.user_col, self.prediction_col], ascending=[True, False])
            .groupby(self.user_col)
            .head(top_k)
            .reset_index(drop=True)
        )