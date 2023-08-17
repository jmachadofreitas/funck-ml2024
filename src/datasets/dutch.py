"""" Dutch Census Dataset """
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional
import itertools

import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets.utils import download_url
import pytorch_lightning as pl
import pandas as pd
from scipy.io.arff import loadarff

from sklearn.pipeline import Pipeline
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


SOURCE_URL = "https://github.com/tailequy/fairness_dataset/raw/main/Dutch_census/dutch_census_2001.arff"
RAW_FILENAME = "dutch_census_2001.arff"

CONTEXT_NAME = "sex"
TARGET_NAME = "occupation"
POSITIVE_OUTCOME = "5_4_9"  # or "2_1"?
RAW_COLUMNS = OrderedDict(
    [
        (CONTEXT_NAME, F(CAT, [2, 1])),
        ("age", F(CAT, [17, 15, 16, 13, 14, 11, 12, 3, 2, 1, 10, 7, 6, 5, 4, 9, 8])),
        (
            "household_position",
            F(CAT, [1110, 1132, 1210, 1131, 1121, 1140, 1122, 1220]),
        ),
        ("household_size", F(CAT, [125, 126, 114, 112, 113, 111])),
        ("prev_residence_place", F(CAT, [2, 1, 9])),
        ("citizenship", F(CAT, [3, 2, 1])),
        ("country_birth", F(CAT, [3, 2, 1])),
        ("edu_level", F(CAT, [3, 2, 1, 0, 5, 4, 9])),
        ("economic_status", F(CAT, [221, 222, 210, 223, 112, 224, 111, 120])),
        (
            "cur_eco_activity",
            F(CAT, [134, 135, 132, 133, 138, 122, 139, 200, 136, 137, 124, 111, 131]),
        ),
        ("Marital_status", F(CAT, [3, 2, 1, 4])),
        (TARGET_NAME, F(CAT, ["5_4_9", "2_1"])),
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
        raise FileNotFoundError(f"{raw_filepath}")


def load_raw_data(dataset_dir) -> Tuple[pd.DataFrame, ...]:
    dataset_dir = Path(dataset_dir)
    raw_data = loadarff(dataset_dir / RAW_FILENAME)
    df = pd.DataFrame(raw_data[0]).stack().str.decode("utf-8").unstack()
    df[TARGET_NAME] = df[TARGET_NAME] == POSITIVE_OUTCOME
    return df


class DutchPreprocessor(object):
    def __init__(
        self,
        dataset_dir: str,
        seed: int = 0,
    ):
        """
        DutchtPreprocessor
        """

        self.dataset_dir = Path(dataset_dir)
        self.context_name = CONTEXT_NAME
        self.base_pkl_filename = "{}.pkl"
        self.seed = seed

        # Splitter
        self.splitter = KFold(n_splits=5)

        # Features
        self.cat_features = OrderedDict(
            [
                (name, feat)
                for name, feat in RAW_COLUMNS.items()
                if feat.type in (BIN, CAT) and name not in (TARGET_NAME, CONTEXT_NAME)
            ]
        )

        # Preprocessing encoders
        self.cat_encoder = OneHotEncoder(
            categories=[f.categories for f in self.cat_features.values()], sparse=False
        )
        # To create numerical feature
        self.num_encoder = Pipeline(
            [
                (
                    "encoder",
                    TargetEncoder(
                        cols=["age"],
                        return_df=False,
                        min_samples_leaf=2,
                        smoothing=0.99,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
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
        df_cat = df.loc[:, [name for name in self.cat_features.keys()]]
        return df_cat, y, c

    def preprocess_train(self, df_cat, y, c):
        df_num = self.num_encoder.fit_transform(df_cat[["age"]], y)
        df_cat = self.cat_encoder.fit_transform(df_cat)
        y = self.target_encoder.fit_transform(y)
        c = self.context_encoder.fit_transform(c)
        return self._to_torch(df_num, df_cat, y, c)

    def preprocess_test(self, df_cat, y, c):
        df_num = self.num_encoder.transform(df_cat[["age"]])
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
        df_cat, y, c = self.split_cols(df_raw)
        tr_idx, te_idx = self.train_test_split(df_cat)
        train = self.preprocess_train(
            df_cat.iloc[tr_idx], y.iloc[tr_idx], c.iloc[tr_idx]
        )
        test = self.preprocess_test(df_cat.iloc[te_idx], y.iloc[te_idx], c.iloc[te_idx])
        return train, test


class Dutch(Dataset):
    def __init__(
        self,
        root="datasets/",
        train=True,
        transform=None,
        target_transform=None,
        seed: int = 0,
    ):
        """Dutch dataset"""
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        self.dataset_dir = Path(root) / "Dutch"
        data_tr, data_te = self.download_preprocess()

        self.data = data_tr if train else data_te
        self.input_shape = (self.data[0].shape[1],)
        self.target_dim = 1
        self.target_type = "bin"
        self.target_probs = [0.52]
        self.context_dim = 1
        self.context_type = "bin"
        self.num_dim = 1
        self.num_idxs = list(range(self.num_dim))
        self.num_var = 1e-4

    def download_preprocess(self):
        """Download, preprocess"""
        filepath = self.dataset_dir / RAW_FILENAME
        if not filepath.exists():
            download_raw_data(self.dataset_dir)
        df_raw = load_raw_data(self.dataset_dir)
        train, test = DutchPreprocessor(dataset_dir=self.dataset_dir, seed=self.seed)(
            df_raw
        )
        return train, test

    @property
    def cat_idxs(self):
        """list with the dim of each categorical variable"""
        out = list()
        offset = self.num_dim
        num_cats = [
            len(f.categories)
            for k, f in RAW_COLUMNS.items()
            if f.type == CAT and k not in (TARGET_NAME, CONTEXT_NAME)
        ]
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


class DutchDataModule(pl.LightningDataModule):
    name = "dutch"
    dataset_cls = Dutch

    def __init__(
        self,
        root: str,
        batch_size: int = 128,
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
                    f"num_samples_per_class is too large: One of the dataloaders is empty."
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


class DutchCommittee(TabularCommittee):
    def __init__(self):
        recon_idx = 0
        data_dict = {
            recon_idx: ("age_num", 1, "num"),
            "y": (TARGET_NAME, 1, "bin"),
            "s": (CONTEXT_NAME, 1, "bin"),
        }
        super().__init__(data_dict, recon_idx)
        self.logger = get_logger(__name__)
