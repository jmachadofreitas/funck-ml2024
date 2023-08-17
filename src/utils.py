from typing import Union, List, Tuple, Sequence
from dataclasses import dataclass
from pathlib import Path
import logging
import random
import traceback
import io

import numpy as np
import pandas as pd
import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf, DictConfig
import hydra

ExperimentConfig = DictConfig


@dataclass
class DatasetConfig:
    input_shape: List[int]
    input_type: str
    target_dim: int
    target_type: str
    context_dim: int
    context_type: str
    num_var: float = 1
    target_name: str = "y"
    context_name: str = "c"
    target_probs: List[float] = None
    num_idxs: List[int] = None
    cat_idxs: List[List[int]] = None


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, pl.utilities.rank_zero_only(getattr(logger, level)))
    return logger


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2**32)
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_model(
    checkpoint_path: Path, dconfig: DatasetConfig, econfig: ExperimentConfig, model=None
):
    if model is None:
        model = hydra.utils.instantiate(econfig.model.target, dconfig, econfig)
    model = model.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path), dconfig=dconfig, econfig=econfig
    )
    return model


def flatten(nested):
    """
    >>> flatten([1, [3, 4, [1, 2, 3], [1], [2, [3, 4]], 2, 3, [3]], [1, 3, 4]])
    """

    def _flatten(n):
        for el in n:
            if isinstance(el, (list, tuple)):
                for e in flatten(el):
                    yield e
            else:
                yield el

    return [el for el in _flatten(nested)]


def parse_sh_array(array, isnum=False):
    array = array.replace("(", "").replace(")", "").split(" ")
    return [eval(el) for el in array] if isnum else array


def dict2csv(results, path):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


def log_traceback(logger, ex, ex_traceback=None):
    if ex_traceback is None:
        ex_traceback = ex.__traceback__
    tb_lines = [
        line.rstrip("\n")
        for line in traceback.format_exception(ex.__class__, ex, ex_traceback)
    ]
    logger.info(tb_lines)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Used?

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    def empty(*args, **kwargs):
        pass

    trainer.logger.log_hyperparams = lambda *args, **kwargs: None
