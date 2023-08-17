"""
ProPublica COMPAS
COMPAS Recidivism Racial Bias
Racial Bias in inmate COMPAS reoffense risk scores for Florida (ProPublica)

References:
    - https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_Lineage_Case_Study
    - https://github.com/burklight/VariationalPrivacyFairness/blob/master/src/utils.py
    - https://www.kaggle.com/danofer/compass
    - https://blog.fastforwardlabs.com/2017/03/09/fairml-auditing-black-box-predictive-models.html
    - Original https://github.com/propublica/compas-analysis/
"""
from pathlib import Path
from typing import Tuple, Optional
from collections import OrderedDict
from filelock import FileLock
import itertools

import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets.utils import download_url
import pytorch_lightning as pl

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)

from .shared import *
from ..utils import get_logger


class OriginalCOMPAS:
    SOURCE_URL = "https://storage.googleapis.com/compas_dataset/"
    RAW_FILENAME = "cox-violent-parsed.csv"
    RAW_COLUMNS = OrderedDict(
        [
            ("age", int),
            ("c_charge_desc", str),
            ("c_charge_degree", str),
            ("c_days_from_compas", int),
            ("juv_fel_count", int),
            ("juv_misd_count", int),
            ("juv_other_count", int),
            ("priors_count", int),
            ("r_days_from_arrest", str),
            ("race", str),
            ("sex", str),
            ("vr_charge_desc", str),
            ("is_recid", str),
        ]
    )
    TARGET_NAME = "is_recid"
    CONTEXT_NAME = "race"

    CATEGORICAL_FEATURE_KEYS = [
        "sex",
        "race",
        "c_charge_desc",
        "c_charge_degree",
    ]

    # List of the unique values for the items within CATEGORICAL_FEATURE_KEYS.
    MAX_CATEGORICAL_FEATURE_VALUES = [2, 6, 513, 14]

    INT_FEATURE_KEYS = [
        "age",
        "c_days_from_compas",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "sample_weight",
    ]

    # Select subset of columns
    # df_raw = df[RAW_COLUMNS]

    # We will use "is_recid" as our ground truth lable, which is boolean value
    # indicating if a defendant committed another crime. There are some rows with -1
    # indicating that there is no data. These rows we will drop from training.
    # df[df["is_recid"] != -1]

    # Given the distribution between races in this dataset we will only focuse on
    # recidivism for African-Americans and Caucasians.
    # df = df[df["race"].isin(["African-American", "Caucasian"])]


class FairMLCOMPAS:
    SOURCE_URL = "https://raw.githubusercontent.com/adebayoj/fairml/master/doc/example_notebooks/"
    RAW_FILENAME = "propublica_data_for_fairml.csv"
    RAW_COLUMNS = OrderedDict(
        [
            ("Number_of_Priors", int),
            ("score_factor", str),
            ("Age_Above_FourtyFive", str),
            ("Age_Below_TwentyFive", str),
            ("African_American", str),
            ("Asian", str),
            ("Hispanic", str),
            ("Native_American", str),
            ("Other", str),
            ("Female", str),
            ("Misdemeanor", str),
            ("Two_yr_Recidivism", str),
        ]
    )
    TARGET_NAME = "Two_yr_Recidivism"
    CONTEXT_NAME = "African_American"


SOURCE_URL = FairMLCOMPAS.SOURCE_URL
RAW_FILENAME = FairMLCOMPAS.RAW_FILENAME
RAW_COLUMNS = FairMLCOMPAS.RAW_COLUMNS
TARGET_NAME = FairMLCOMPAS.TARGET_NAME
CONTEXT_NAME = FairMLCOMPAS.TARGET_NAME


def download_raw_data(dataset_dir):
    raw_filepath = Path(dataset_dir) / RAW_FILENAME
    if not raw_filepath.exists():
        download_url(
            SOURCE_URL + RAW_FILENAME, root=str(dataset_dir), filename=RAW_FILENAME
        )
    if not raw_filepath.exists():
        raise FileNotFoundError(f"{raw_filepath}")


def load_raw_data(dataset_dir) -> pd.DataFrame:
    df = pd.read_csv(Path(dataset_dir) / RAW_FILENAME, delimiter=",")
    return df[RAW_COLUMNS.keys()]


class FairMLCOMPASPreprocessor(object):
    def __init__(self, dataset_dir: Path, seed: int = 0, cache: bool = False):
        """
        FairMLCOMPASPreprocessor
        """
        self.target_name = FairMLCOMPAS.TARGET_NAME
        self.context_name = FairMLCOMPAS.CONTEXT_NAME
        self.raw_columns = FairMLCOMPAS.RAW_COLUMNS

        self.dataset_dir = Path(dataset_dir)
        # self.test_prop = 0.2

        self.base_pkl_filename = "{}.pkl"
        self.seed = seed
        self.cache = cache

        # Splitter
        self.splitter = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        # self.splitter = ShuffleSplit(test_size=self.test_prop, random_state=self.seed)

        # Preprocessing encoders
        self.input_scaler = StandardScaler()
        # self.input_scaler = MinMaxScaler()
        self.context_encoder = LabelEncoder()
        self.target_encoder = LabelEncoder()

    def train_test_split(self, df):
        num_splits = self.splitter.get_n_splits(df)
        split_idx = self.seed % num_splits
        gen = self.splitter.split(df)
        tr_idxs, te_idxs = next(itertools.islice(gen, split_idx, None))
        return df.iloc[tr_idxs], df.iloc[te_idxs]

    def split_cols(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cat = df.loc[:, [name for name, dtype in RAW_COLUMNS.items() if dtype == str]]
        num = df.loc[:, [name for name, dtype in RAW_COLUMNS.items() if dtype == int]]
        y = cat.pop(self.target_name)
        c = cat.pop(self.context_name)
        return num, cat, y, c

    def preprocess_train(self, df):
        num, cat, y, c = self.split_cols(df)
        num = self.input_scaler.fit_transform(num)
        cat = cat.to_numpy()
        y = self.target_encoder.fit_transform(y)
        c = self.context_encoder.fit_transform(c)
        return self._to_torch(num, cat, y, c)

    def preprocess_test(self, df):
        num, cat, y, c = self.split_cols(df)
        num = self.input_scaler.transform(num)
        cat = cat.to_numpy()
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


class COMPAS(Dataset):
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

        self.dataset_dir = Path(root) / "COMPAS"
        data_tr, data_te = self.download_preprocess()

        self.data = data_tr if train else data_te
        self.input_shape = (self.data[0].shape[1],)
        self.target_dim = 1
        self.target_type = "bin"
        self.target_probs = [0.45]
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
        raw_data = load_raw_data(self.dataset_dir)
        train, test = FairMLCOMPASPreprocessor(
            dataset_dir=self.dataset_dir, seed=self.seed
        )(raw_data)
        return train, test

    @property
    def cat_idxs(self):
        """list with the dim of each categorical variable"""
        out = list()
        offset = self.num_dim
        num_cats = [1 for _ in range(9)]
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
        )


class COMPASDataModule(pl.LightningDataModule):
    name = "compas"
    dataset_cls = COMPAS

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
            ):  # Compas is too small for number of labels
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


class COMPASCommittee(TabularCommittee):
    def __init__(self):
        recon_idx = 0
        data_dict = {
            recon_idx: ("Number_of_Priors", 1, "num"),
            "y": ("Two_yr_Recidivism", 1, "bin"),
            "s": ("African_American", 1, "bin"),
        }
        super().__init__(data_dict)
        self.logger = get_logger(__name__)
