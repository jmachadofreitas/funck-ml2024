from typing import Union
from collections import defaultdict, OrderedDict
import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


def dataloader2numpy(dataloader):
    batches = {key: list() for key in ["x", "y", "s"]}
    for batch in dataloader:
        batches["x"].append(batch[0])
        batches["y"].append(batch[1])
        batches["s"].append(batch[2])

    x = torch.cat(batches["x"]).numpy()
    y = torch.cat(batches["y"]).numpy()
    s = torch.cat(batches["s"]).numpy()
    return x, y, s


@torch.no_grad()
def get_representations_and_numpy_dataset(model, dataloader):
    model.eval()
    batches = {key: list() for key in ["z", "x", "y", "s"]}
    for batch in dataloader:
        z, (x, y, s) = model(batch[0], batch[2]), batch
        batches["z"].append(z)
        batches["x"].append(x)
        batches["y"].append(y)
        batches["s"].append(s)
    z = torch.cat(batches["z"]).numpy()
    x = torch.cat(batches["x"]).numpy()
    y = torch.cat(batches["y"]).numpy()
    s = torch.cat(batches["s"]).numpy()
    return z, (x, y, s)


@torch.no_grad()
def get_representations_and_torch_dataset(model, dataloader):
    model.eval()
    batches = {key: list() for key in ["z", "x", "y", "s"]}
    for batch in dataloader:
        z, (x, y, s) = model(batch[0], batch[2]), batch
        batches["z"].append(z)
        batches["x"].append(x)
        batches["y"].append(y)
        batches["s"].append(s)
    z = torch.cat(batches["z"])
    x = torch.cat(batches["x"])
    y = torch.cat(batches["y"])
    s = torch.cat(batches["s"])
    return z, (x, y, s)


def numpy2dataloader(x, y, s):
    """Necessary ?"""
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    s = torch.Tensor(s)
    dataset = TensorDataset(x, y, s)
    return DataLoader(dataset)


def reshape_results(results, metadata=None):
    """TODO"""
    if metadata is None:
        metadata = dict()

    output = list()
    for k, v in results.items():  # Iterate metrics
        new_entry = dict()
        new_entry["metric"] = k
        new_entry["value"] = v
        new_entry.update(metadata)
        output.append(new_entry)
    return output


def get_z(model, x):
    model.eval()
    with torch.no_grad():
        if model.device == torch.device("cuda:0"):
            x = torch.Tensor(x).cuda()
        else:
            x = torch.Tensor(x)
        z = model(x)
    model.train()
    return z.detach().numpy()


def generate_representations(model, dataloader, metadata=None):
    if metadata is None:
        metadata = dict()
    else:
        metadata = metadata.copy()

    # General parameters
    params = {param: getattr(model.hparams, param, None) for param in ["beta"]}

    output = defaultdict(list)
    for batch in dataloader:
        x, y = batch
        z = get_z(model, x)
        output[f"z{idx}"].append(z[:, idx])
        output[f"y{idx}"].append(y[:, idx].reshape(-1).numpy())

    return embeds


def save_representations(embeds: Tensor, metadata: dict = None):
    # Create column names
    batch_size, latent_dim = embeds.shape
    columns_names = [f"z{idx}" for idx in range(latent_dim)]
    columns_names.append("y")

    output = dict()
    # embeds.numpy(), column_names=columns_names

    # output = {k: utils.flatten(v) for k, v in output.items()}
    # output.update(metadata)
    return output
