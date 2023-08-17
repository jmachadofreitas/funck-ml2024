""" Credit Card Clients """
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional
from filelock import FileLock
import itertools
from dataclasses import dataclass
from enum import Enum, auto

import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets.utils import download_url
from pytorch_lightning import LightningDataModule
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)
from category_encoders import TargetEncoder

from ..utils import get_logger
from .shared import *


SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
RAW_FILENAME = "default of credit card clients.xls"


CONTEXT_NAME = "SEX"
TARGET_NAME = "default payment next month"
POSITIVE_OUTCOME = 1
RAW_COLUMNS = OrderedDict(
    [
        ("ID", F(ID)),
        ("LIMIT_BAL", F(NUM)),
        (CONTEXT_NAME, F(BIN, [1, 2])),  # Context
        ("EDUCATION", F(CAT, [0, 1, 2, 3, 4, 5, 6])),
        ("MARRIAGE", F(CAT, [0, 1, 2, 3])),
        ("AGE", F(NUM)),
        ("PAY_0", F(NUM)),
        ("PAY_2", F(NUM)),
        ("PAY_3", F(NUM)),
        ("PAY_4", F(NUM)),
        ("PAY_5", F(NUM)),
        ("PAY_6", F(NUM)),
        ("BILL_AMT1", F(NUM)),
        ("BILL_AMT2", F(NUM)),
        ("BILL_AMT3", F(NUM)),
        ("BILL_AMT4", F(NUM)),
        ("BILL_AMT5", F(NUM)),
        ("BILL_AMT6", F(NUM)),
        ("PAY_AMT1", F(NUM)),
        ("PAY_AMT2", F(NUM)),
        ("PAY_AMT3", F(NUM)),
        ("PAY_AMT4", F(NUM)),
        ("PAY_AMT5", F(NUM)),
        ("PAY_AMT6", F(NUM)),
        (TARGET_NAME, F(BIN)),
    ]
)


def download_raw_data(dataset_dir):
    # TODO: Unzip ...
    raw_filepath = dataset_dir / RAW_FILENAME
    if not raw_filepath.exists():
        download_url(
            SOURCE_URL + RAW_FILENAME, root=str(dataset_dir), filename=RAW_FILENAME
        )
    if not raw_filepath.exists():
        raise FileNotFoundError(f"{train_raw_filepath}")


def load_raw_data(dataset_dir) -> Tuple[pd.DataFrame, ...]:
    """101767 - 98054 = 3713"""
    dataset_dir = Path(dataset_dir)
    df = pd.read_excel(dataset_dir / RAW_FILENAME, skiprows=1)
    # if CONTEXT_NAME == "AGE":
    #     df[CONTEXT_NAME] = df[CONTEXT_NAME] >= 30
    df[TARGET_NAME] = df[TARGET_NAME] == POSITIVE_OUTCOME
    return df


class CreditPreprocessor(object):
    def __init__(
        self,
        dataset_dir: str,
        seed: int = 0,
    ):
        """
        CreditPreprocessor

        Args:
            root: datamodule root
        """

        self.dataset_dir = Path(dataset_dir)
        self.context_name = CONTEXT_NAME
        self.base_pkl_filename = "{}.pkl"
        self.seed = seed

        # Splitter
        self.splitter = KFold(n_splits=5)

        # Features
        self.num_features = OrderedDict(
            [
                (name, feat)
                for name, feat in RAW_COLUMNS.items()
                if feat.type == NUM and name not in (TARGET_NAME, CONTEXT_NAME)
            ]
        )
        self.cat_features = OrderedDict(
            [
                (name, feat)
                for name, feat in RAW_COLUMNS.items()
                if feat.type in (BIN, CAT) and name not in (TARGET_NAME, CONTEXT_NAME)
            ]
        )

        # Preprocessing encoders
        self.num_encoder = StandardScaler()
        self.cat_encoder = OneHotEncoder(
            categories=[f.categories for f in self.cat_features.values()], sparse=False
        )
        self.context_encoder = LabelEncoder()
        self.target_encoder = LabelEncoder()

    def train_test_split(self, x):
        num_splits = self.splitter.get_n_splits(x)
        split_idx = self.seed % num_splits
        gen = self.splitter.split(x)
        tr_idxs, te_idxs = next(itertools.islice(gen, split_idx, None))
        return tr_idxs, te_idxs

    def split_cols(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Split variables"""
        y = df.pop(TARGET_NAME)
        c = df.pop(CONTEXT_NAME)
        df_num = df.loc[:, [name for name in self.num_features.keys()]]
        df_cat = df.loc[:, [name for name in self.cat_features.keys()]]
        return df_num, df_cat, y, c

    def preprocess_train(self, df_num, df_cat, y, c):
        df_num = self.num_encoder.fit_transform(df_num)
        df_cat = self.cat_encoder.fit_transform(df_cat)
        y = self.target_encoder.fit_transform(y)
        c = self.context_encoder.fit_transform(c)
        return self._to_torch(df_num, df_cat, y, c)

    def preprocess_test(self, df_num, df_cat, y, c):
        df_num = self.num_encoder.transform(df_num)
        df_cat = self.cat_encoder.transform(df_cat)
        y = self.target_encoder.transform(y)
        c = self.context_encoder.transform(c)
        return self._to_torch(df_num, df_cat, y, c)

    @staticmethod
    def _to_torch(df_num, df_cat, y, c):
        x = torch.cat(
            [torch.from_numpy(df_num).float(), torch.from_numpy(df_cat).float()], dim=1
        )
        y = torch.from_numpy(y).long()
        c = torch.from_numpy(c).long()
        return x, y, c

    @staticmethod
    def save(filename, data):
        if not Path(filename).exists():
            torch.save(data, filename)

    def __call__(self, df_raw):
        df_num, df_cat, y, c = self.split_cols(df_raw)
        tr_idx, te_idx = self.train_test_split(df_num)
        train = self.preprocess_train(
            df_num.iloc[tr_idx], df_cat.iloc[tr_idx], y.iloc[tr_idx], c.iloc[tr_idx]
        )
        test = self.preprocess_test(
            df_num.iloc[te_idx], df_cat.iloc[te_idx], y.iloc[te_idx], c.iloc[te_idx]
        )
        return train, test


class Credit(Dataset):
    def __init__(
        self,
        root="datasets/",
        train=True,
        transform=None,
        target_transform=None,
        seed: int = 0,
    ):
        """Credit dataset"""
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        self.dataset_dir = Path(root) / "Credit"
        data_tr, data_te = self.download_preprocess()

        self.data = data_tr if train else data_te
        self.input_shape = (self.data[0].shape[1],)
        self.target_dim = 1
        self.target_type = "bin"
        self.target_probs = [0.22]
        self.context_dim = 1
        self.context_type = "bin"
        self.num_dim = 20
        self.num_idxs = list(range(self.num_dim))
        self.num_var = 1e-3

    def download_preprocess(self):
        """Download, preprocess"""
        filepath = self.dataset_dir / RAW_FILENAME
        if not filepath.exists():
            download_raw_data(self.dataset_dir)
        df_raw = load_raw_data(self.dataset_dir)
        train, test = CreditPreprocessor(dataset_dir=self.dataset_dir, seed=self.seed)(
            df_raw
        )
        return train, test

    @property
    def cat_idxs(self):
        """list with the dim of each categorical variable"""
        out = list()
        offset = self.num_dim
        num_cats = [len(f.categories) for k, f in RAW_COLUMNS.items() if f.type == CAT]
        for n in num_cats:
            out.append((offset, offset + n))
            offset += n
        return out

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        obs, target, context = self.data[0][idx], self.data[1][idx], self.data[2][idx]
        if self.transform is not None:
            obs = self.transform(obs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return obs, target, context

    def get_model_kwargs(self):
        return dict(
            input_shape=self.input_shape,
            input_type="mix",
            target_dim=self.target_dim,
            target_type=self.target_type,
            target_probs=self.target_probs,
            context_dim=self.context_dim,
            context_type=self.context_type,
            num_idxs=self.num_idxs,
            cat_idxs=self.cat_idxs,
            num_var=self.num_var,
        )


class CreditDataModule(LightningDataModule):
    name = "credit"
    dataset_cls = Credit

    def __init__(
        self,
        root: str = "datasets/",
        batch_size: int = 32,
        val_prop: float = 0.1,
        seed: int = 12345,
        num_samples_per_class: int = -1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size
        self.datasets = dict()
        self.numpy_datasets = dict()
        self.val_prop = val_prop
        self.seed = seed

        self.ds_kwargs = dict(root=root, seed=seed)
        self.dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create preprocessed datamodule for all splits"""

        # Train/Val Split
        data_tmp = self.dataset_cls(train=True, **self.ds_kwargs)
        tmp_len = len(data_tmp)
        train_len = round((1 - self.val_prop) * tmp_len)
        lengths = [train_len, tmp_len - train_len]
        self.datasets["train"], self.datasets["val"] = random_split(data_tmp, lengths)

        # Test Split
        self.datasets["test"] = self.dataset_cls(train=False, **self.ds_kwargs)

    def train_dataloader(self):
        if self.num_samples_per_class > 0:
            labeled_sampler, unlabeled_sampler = get_semisup_samplers(
                self.datasets["train"],
                num_samples_per_class=self.num_samples_per_class,
                num_classes=2,
            )
            if len(labeled_sampler) > 0 and len(unlabeled_sampler) > 0:
                return (
                    DataLoader(
                        self.datasets["train"],
                        sampler=labeled_sampler,
                        **self.dl_kwargs,
                    ),
                    DataLoader(
                        self.datasets["train"],
                        sampler=unlabeled_sampler,
                        **self.dl_kwargs,
                    ),
                )
            else:
                raise ValueError(
                    f"'num_samples_per_class' is too large: One of the dataloaders is empty."
                )
        else:
            return DataLoader(self.datasets["train"], shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], **self.dl_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.datasets["test"], **self.dl_kwargs)

    def get_model_kwargs(self):
        if "train" not in self.datasets:
            self.setup()
        return self.datasets["train"].dataset.get_model_kwargs()


class CreditCommittee(TabularCommittee):
    def __init__(self):
        recon_idx = 0
        data_dict = {
            recon_idx: ("LIMIT_BAL", 1, "num"),
            "y": (TARGET_NAME, 1, "bin"),
            "s": (CONTEXT_NAME, 1, "bin"),
        }
        super().__init__(data_dict)
        self.logger = get_logger(__name__)
