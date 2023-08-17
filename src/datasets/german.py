"""
German dataset

Target variable:
    Credit score (cr_good_bad)

Protected attributes:
    age: binarized
        age > 65:  Zemel et al. (2013) Learning Fair Representations
        age > 25: Moyer et al. (2019) Invariant Representations without Adversarial Training
    sex: Louizos et al. (2017) The Variational Fair Autoencoder  <--
"""
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional
import itertools
from filelock import FileLock

import torch
from torch.utils.data import random_split, Dataset, ConcatDataset, DataLoader
from torchvision.datasets.utils import download_url
import pytorch_lightning as pl

import pandas as pd
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)

from ..utils import get_logger
from .shared import *


SOURCE_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
RAW_FILENAME = "german.data"

RAW_COLUMNS = OrderedDict(
    [
        ("checking_acc", str),  #
        ("duration", int),
        ("credit_hist", str),  #
        ("purpose", str),  #
        ("credit_amount", int),
        ("savings", str),  #
        ("employment_status", str),  #
        ("install_rate", int),
        ("relationship_and_sex", str),  #
        ("debtors", str),  #
        ("res_interval", int),
        ("property", str),  #
        ("age", str),  # S (binarized age>val)
        ("other_plans", str),  #
        ("housing", str),  #
        ("credits_at_bank", int),
        ("job", str),  #
        ("liable_persons", int),
        ("phone", str),  #
        ("foreign", str),  #
        ("cr_good_bad", str),  # T
    ]
)

CONTEXT_NAME = "age"
TARGET_NAME = "cr_good_bad"
POSITIVE_OUTCOME = "1"

CATEGORIES = OrderedDict(
    [
        ("checking_acc", ["A11", "A12", "A13", "A14"]),
        ("credit_hist", ["A30", "A31", "A32", "A33", "A34"]),
        ("purpose", [f"A{i}" for i in range(40, 50)] + ["A410"]),
        ("savings", [f"A{i}" for i in range(61, 66)]),
        ("employment_status", [f"A{i}" for i in range(71, 76)]),
        ("relationship_and_sex", [f"A{i}" for i in range(91, 96)]),
        ("debtors", ["A101", "A102", "A103"]),
        ("property", [f"A{i}" for i in range(121, 125)]),
        # ("age", str),  # S (binarized >65)
        ("other_plans", ["A141", "A142", "A143"]),
        ("housing", ["A151", "A152", "A153"]),
        ("job", ["A171", "A172", "A173", "A174"]),
        ("phone", ["A191", "A192"]),
        ("foreign", ["A201", "A202"]),
        # ("cr_good_bad", str)  # T
        # ("sex", [False, True])  # S (Alt.)
    ]
)


def download_raw_data(dataset_dir):
    raw_filepath = dataset_dir / RAW_FILENAME
    if not raw_filepath.exists():
        download_url(
            SOURCE_URL + RAW_FILENAME, root=str(dataset_dir), filename=RAW_FILENAME
        )
    if not raw_filepath.exists():
        raise FileNotFoundError(f"{raw_filepath}")


def load_raw_data(dataset_dir) -> pd.DataFrame:
    df = pd.read_csv(
        dataset_dir / RAW_FILENAME,
        names=[name for name in RAW_COLUMNS.keys()],
        dtype=RAW_COLUMNS,
        delimiter=" ",
        na_values="?",
        keep_default_na=False,
    )
    df.dropna(inplace=True)
    return df


class GermanPreprocessor(object):
    def __init__(self, dataset_dir: Path, seed: int = 0, cache: bool = False):
        """
        GermanPreprocessor

        Only implemented for sensitive attribute age (>50).

        """
        self.dataset_dir = Path(dataset_dir)
        self.context_name = "age"
        self.context_thresh = 25  # From arXiv:2110.00530
        self.base_pkl_filename = "{}.pkl"
        self.seed = seed
        self.cache = cache
        # self.test_prop = 0.2

        # Splitter
        self.splitter = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.seed
        )
        # self.splitter = StratifiedShuffleSplit(test_size=self.test_prop, random_state=self.seed)
        # self.splitter = ShuffleSplit(test_size=self.test_prop, random_state=self.seed)

        # Preprocessing encoders
        self.input_scaler = StandardScaler()
        # self.input_scaler = MinMaxScaler()
        self.input_encoder = OneHotEncoder(
            categories=[v for v in CATEGORIES.values()], sparse=False
        )
        # input_encoder = OrdinalEncoder()

        self.context_encoder = LabelEncoder()
        self.target_encoder = LabelEncoder()

    @staticmethod
    def _preprocess_target(target: pd.Series):
        return target == POSITIVE_OUTCOME  # 1 = Good, 2 = Bad

    def _preprocess_context(self, context: pd.Series):
        if self.context_name == "age":
            return context.astype(int) <= self.context_thresh
        else:  # self.context_name == "relationship_and_sex"
            return (context == "A92") | (context == "A95")  # Female

    def train_test_split(self, df):
        x, y = df, self._preprocess_target(df[TARGET_NAME])
        num_splits = self.splitter.get_n_splits(x, y)
        split_idx = self.seed % num_splits
        gen = self.splitter.split(x, y)
        tr_idxs, te_idxs = next(itertools.islice(gen, split_idx, None))
        return df.iloc[tr_idxs], df.iloc[te_idxs]

    def split_cols(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        num = df.loc[:, [name for name, dtype in RAW_COLUMNS.items() if dtype == int]]
        cat = df.loc[:, [name for name, dtype in RAW_COLUMNS.items() if dtype == str]]
        y = self._preprocess_target(df[TARGET_NAME])
        if self.context_name == "age":
            c = self._preprocess_context(df["age"])
        elif self.context_name == "relationship_and_sex":
            c = self._preprocess_context(df["relationship_and_sex"])
        else:
            raise NotImplementedError
        cat.drop(columns=[TARGET_NAME, self.context_name], inplace=True)
        return num, cat, y, c

    def preprocess_train(self, df):
        num, cat, y, c = self.split_cols(df)
        num = self.input_scaler.fit_transform(num)
        cat = self.input_encoder.fit_transform(cat)
        y = self.target_encoder.fit_transform(y)
        c = self.context_encoder.fit_transform(c)
        return self._to_torch(num, cat, y, c)

    def preprocess_test(self, df):
        num, cat, y, c = self.split_cols(df)
        num = self.input_scaler.transform(num)
        cat = self.input_encoder.transform(cat)
        y = self.target_encoder.transform(y)
        c = self.context_encoder.transform(c)
        return self._to_torch(num, cat, y, c)

    @staticmethod
    def _to_torch(num, cat, y, c):
        x = torch.cat(
            [torch.from_numpy(num).float(), torch.from_numpy(cat).float()], dim=1
        )
        y = torch.from_numpy(y).long()
        c = torch.from_numpy(c).long()
        return x, y, c

    @staticmethod
    def save(filename, data):
        if not Path(filename).exists():
            torch.save(data, filename)

    def __call__(self, df_raw):
        df_tr, df_te = self.train_test_split(df_raw)

        train = self.preprocess_train(df_tr)
        if self.cache:
            filename = self.dataset_dir / self.base_pkl_filename.format("train")
            self.save(filename, train)

        test = self.preprocess_train(df_te)
        if self.cache:
            filename = self.dataset_dir / self.base_pkl_filename.format("test")
            self.save(filename, test)

        return train, test


class German(Dataset):
    def __init__(
        self,
        root="datamodule/",
        train=True,
        transform=None,
        target_transform=None,
        seed: int = 0,
    ):
        """German dataset"""
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        self.dataset_dir = Path(root) / "German"
        data_tr, data_te = self.download_preprocess()

        self.data = data_tr if train else data_te
        self.input_shape = (self.data[0].shape[1],)
        self.target_dim = 1
        self.target_type = "bin"
        self.target_probs = [0.68]
        self.context_dim = 1
        self.context_type = "bin"
        self.num_dim = 6
        self.num_idxs = list(range(self.num_dim))
        self.num_var = 1e-4

    def download_preprocess(self):
        """Download, preprocess"""
        filepath = self.dataset_dir / RAW_FILENAME
        if not filepath.exists():
            download_raw_data(self.dataset_dir)
        raw_data = load_raw_data(self.dataset_dir)
        train, test = GermanPreprocessor(dataset_dir=self.dataset_dir, seed=self.seed)(
            raw_data
        )
        return train, test

    @property
    def cat_idxs(self):
        """list with the dim of each categorical variable"""
        out = list()
        offset = self.num_dim
        num_cats = [len(v) for v in CATEGORIES.values()]
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


class GermanDataModule(pl.LightningDataModule):
    name = "german"
    dataset_cls = German

    def __init__(
        self,
        root: str,
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
            elif (
                len(unlabeled_sampler) == 0
            ):  # German is too small for number of labels
                return DataLoader(
                    self.datasets["train"], sampler=labeled_sampler, **self.dl_kwargs
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


class GermanCommittee(TabularCommittee):
    def __init__(self):
        data_dict = {
            0: ("duration", 1, "num"),
            "y": ("cr_good_bad", 1, "bin"),
            "s": ("age", 1, "bin"),
        }
        super().__init__(data_dict)
        self.logger = get_logger(__name__)
