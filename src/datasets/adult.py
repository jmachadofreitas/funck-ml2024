"""
Adult dataset

Target variable:
    Bank balance >50k (income):
        Louizos et al. (2017) The Variational Fair Autoencoder

Protected attributes:
    sex: <--
        Moyer et al. (2019) Invariant Representations without Adversarial Training
        Zemel et al. (2013) Learning Fair Representations
    age:
        Louizos et al. (2017) The Variational Fair Autoencoder - Doesn"t say how it was binarized
"""
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional
from filelock import FileLock
import itertools

import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets.utils import download_url
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)

from ..utils import get_logger
from .shared import *

SOURCE_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
TRAIN_RAW_FILENAME = "adult.data"
TEST_RAW_FILENAME = "adult.test"

RAW_COLUMNS = OrderedDict(
    [
        ("age", int),  #
        ("workclass", str),  #
        ("fnlwgt", int),
        ("education", str),  #
        ("education-num", int),
        ("marital-status", str),  #
        ("occupation", str),  #
        ("relationship", str),  #
        ("race", str),  #
        ("sex", str),  # S <-
        ("capital-gain", int),
        ("capital-loss", int),
        ("hours-per-week", int),
        ("native-country", str),  #
        ("income", str),  # T <-
    ]
)
POSITIVE_OUTCOME = ">50K"

CATEGORIES = OrderedDict(
    [
        # "age": "continuous",
        (
            "workclass",
            [
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked",
            ],
        ),
        # "fnlwgt": "continuous",
        (
            "education",
            [
                "Bachelors",
                "Some-college",
                "11th",
                "HS-grad",
                "Prof-school",
                "Assoc-acdm",
                "Assoc-voc",
                "9th",
                "7th-8th",
                "12th",
                "Masters",
                "1st-4th",
                "10th",
                "Doctorate",
                "5th-6th",
                "Preschool",
            ],
        ),
        # "education-num": "continuous",
        (
            "marital-status",
            [
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
                "Separated",
                "Widowed",
                "Married-spouse-absent",
                "Married-AF-spouse",
            ],
        ),
        (
            "occupation",
            [
                "Tech-support",
                "Craft-repair",
                "Other-service",
                "Sales",
                "Exec-managerial",
                "Prof-specialty",
                "Handlers-cleaners",
                "Machine-op-inspct",
                "Adm-clerical",
                "Farming-fishing",
                "Transport-moving",
                "Priv-house-serv",
                "Protective-serv",
                "Armed-Forces",
            ],
        ),
        (
            "relationship",
            [
                "Wife",
                "Own-child",
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Unmarried",
            ],
        ),
        (
            "race",
            ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ),
        # ("sex", ["Female", "Male"]),
        # "capital-gain": "continuous",
        # "capital-loss": "continuous",
        # "hours-per-week": "continuous",
        (
            "native-country",
            [
                "United-States",
                "Cambodia",
                "England",
                "Puerto-Rico",
                "Canada",
                "Germany",
                "Outlying-US(Guam-USVI-etc)",
                "India",
                "Japan",
                "Greece",
                "South",
                "China",
                "Cuba",
                "Iran",
                "Honduras",
                "Philippines",
                "Italy",
                "Poland",
                "Jamaica",
                "Vietnam",
                "Mexico",
                "Portugal",
                "Ireland",
                "France",
                "Dominican-Republic",
                "Laos",
                "Ecuador",
                "Taiwan",
                "Haiti",
                "Columbia",
                "Hungary",
                "Guatemala",
                "Nicaragua",
                "Scotland",
                "Thailand",
                "Yugoslavia",
                "El-Salvador",
                "Trinadad&Tobago",
                "Peru",
                "Hong",
                "Holand-Netherlands",
            ],
        ),
    ]
)


def download_raw_data(dataset_dir):
    train_raw_filepath = dataset_dir / TRAIN_RAW_FILENAME
    test_raw_filepath = dataset_dir / TEST_RAW_FILENAME
    if not train_raw_filepath.exists():
        download_url(
            SOURCE_URL + TRAIN_RAW_FILENAME,
            root=str(dataset_dir),
            filename=TRAIN_RAW_FILENAME,
        )
    if not test_raw_filepath.exists():
        download_url(
            SOURCE_URL + TEST_RAW_FILENAME,
            root=str(dataset_dir),
            filename=TEST_RAW_FILENAME,
        )
    if not train_raw_filepath.exists() or not test_raw_filepath.exists():
        raise FileNotFoundError(f"{train_raw_filepath} - {test_raw_filepath}")


def load_raw_data(dataset_dir) -> Tuple[pd.DataFrame, ...]:
    df_tr = pd.read_csv(
        dataset_dir / TRAIN_RAW_FILENAME,
        names=[name for name in RAW_COLUMNS.keys()],
        dtype=RAW_COLUMNS,
        delimiter=", ",
        na_values="?",
        keep_default_na=False,
        engine="python",
    )
    df_te = pd.read_csv(
        dataset_dir / TEST_RAW_FILENAME,
        names=[name for name in RAW_COLUMNS.keys()],
        dtype=RAW_COLUMNS,
        delimiter=", ",
        na_values="?",
        keep_default_na=False,
        skiprows=1,
        engine="python",
    )
    df_tr["income"] = df_tr["income"].str.strip(" .")
    df_te["income"] = df_te["income"].str.strip(" .")
    df_tr.dropna(inplace=True)
    df_te.dropna(inplace=True)
    return df_tr, df_te


class AdultPreprocessor(object):
    def __init__(self, dataset_dir: str, seed: int = 0, cache: bool = False):
        """
        AdultPreprocessor

        Only implemented for sensitive attribute sex.

        Args:
            root: datamodule root
        """
        self.dataset_dir = Path(dataset_dir)
        self.context_name = "sex"
        self.base_pkl_filename = "{}.pkl"
        self.seed = seed
        self.cache = cache
        # self.test_prop = 0.2

        # Splitter
        self.splitter = KFold(n_splits=5, shuffle=True, random_state=self.seed)
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

    def train_test_split(self, df):
        num_splits = self.splitter.get_n_splits(df)
        split_idx = self.seed % num_splits
        gen = self.splitter.split(df)
        tr_idxs, te_idxs = next(itertools.islice(gen, split_idx, None))
        return df.iloc[tr_idxs], df.iloc[te_idxs]

    def split_cols(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Split variables"""
        df = df.assign(sex=(df["sex"] == "Female"))
        cat = df.loc[:, [name for name, dtype in RAW_COLUMNS.items() if dtype == str]]
        num = df.loc[:, [name for name, dtype in RAW_COLUMNS.items() if dtype != str]]
        y = cat.pop("income") == POSITIVE_OUTCOME
        c = cat.pop(self.context_name)
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

    def __call__(self, df_raw_tr, df_raw_te):
        df_raw = pd.concat([df_raw_tr, df_raw_te])
        df_tr, df_te = self.train_test_split(df_raw)

        train = self.preprocess_train(df_tr)
        if self.cache:
            filename = self.dataset_dir / self.base_pkl_filename.format("train")
            self.save(filename, train)

        test = self.preprocess_test(df_te)
        if self.cache:
            filename = self.dataset_dir / self.base_pkl_filename.format("test")
            self.save(filename, test)

        return train, test


class Adult(Dataset):
    def __init__(
        self,
        root="datasets/",
        train=True,
        transform=None,
        target_transform=None,
        seed: int = 0,
    ):
        """Adult dataset"""
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        self.dataset_dir = Path(root) / "Adult"
        data_tr, data_te = self.download_preprocess()

        self.data = data_tr if train else data_te
        self.input_shape = (self.data[0].shape[1],)
        self.target_dim = 1
        self.target_type = "bin"
        self.target_probs = [0.25]
        self.context_dim = 1
        self.context_type = "bin"
        self.num_dim = 6
        self.num_idxs = list(range(self.num_dim))
        self.num_var = 1e-4

    def download_preprocess(self):
        """Download, preprocess"""
        filepaths = [
            self.dataset_dir / filename
            for filename in (TRAIN_RAW_FILENAME, TEST_RAW_FILENAME)
        ]
        if not all(fp.exists() for fp in filepaths):
            download_raw_data(self.dataset_dir)
        train_raw, test_raw = load_raw_data(self.dataset_dir)
        train, test = AdultPreprocessor(dataset_dir=self.dataset_dir, seed=self.seed)(
            train_raw, test_raw
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


class AdultDataModule(pl.LightningDataModule):
    name = "adult"
    dataset_cls = Adult

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


class AdultCommittee(TabularCommittee):
    def __init__(self):
        recon_idx = 0
        data_dict = {
            recon_idx: ("age", 1, "num"),
            "y": ("income", 1, "bin"),
            "s": ("sex", 1, "bin"),
        }
        super().__init__(data_dict, recon_idx)
        self.logger = get_logger(__name__)
