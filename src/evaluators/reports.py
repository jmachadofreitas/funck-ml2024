"""
Reporting utilities
"""
from pathlib import Path
import json
from collections import OrderedDict
from typing import Union, Tuple, Sequence

from tinydb import TinyDB, Query
from tinydb.queries import QueryInstance
import pandas as pd
import numpy as np

from .evaluators import get_representations_and_numpy_dataset
from .. import utils

Records = Sequence[dict]


def create_metadata(econfig, model, datamodule, checkpoint_path, ckpt_type):
    metadata = OrderedDict()

    for p in ["alpha", "beta", "gamma", "delta"]:
        if hasattr(model, p):
            metadata[p] = getattr(model, p)

    metadata.update(
        OrderedDict(
            dataset=econfig.datamodule.name,
            datamodule=econfig.datamodule.target._target_,
            num_samples_per_class=econfig.datamodule.num_samples_per_class,
            model=econfig.model.name,
            plmodule=econfig.model.target._target_,
            batch_size=datamodule.batch_size,
            latent_dim=econfig.model.latent_dim,
            seed=econfig.seed,
            ckpt_type=ckpt_type,
            checkpoint_path=str(checkpoint_path),
        )
    )
    return metadata


def create_results_json(metadata, results, flatten=False):
    if flatten:
        results = utils.flatten(results)
    output = OrderedDict(
        metadata=metadata,
        results=results,
    )
    obj = json.dumps(output, indent=2, sort_keys=False)
    return obj


def save_results_json(filepath, metadata, results, flatten=False):
    if flatten:
        results = utils.flatten(results)
    output = OrderedDict(metadata=metadata, results=results)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, sort_keys=False)


def records2dataframe(
    metadata: Sequence[dict], results: Sequence[dict]
) -> pd.DataFrame:
    """Expand experiment results (JSON) to pd.DataFrame"""
    metadata = pd.DataFrame.from_records(metadata)
    results = pd.DataFrame.from_records(results)
    records = [
        {"idx": idx, "metric": k, "value": v}
        for idx, m in enumerate(results.pop("metric"))
        for k, v in m.items()
    ]
    metrics = pd.DataFrame.from_records(records, index="idx")
    results = results.join(metrics)
    return metadata.merge(results, on="uuid")


def save_representations(
    experiment_dir: Path,
    checkpoint_path: Path,
    model,
    dataloader,
):
    """Save representations"""
    experiment_dir = Path(experiment_dir)
    checkpoint_path = Path(checkpoint_path)
    z, (x, y, s) = get_representations_and_numpy_dataset(model, dataloader)
    ckpt_type = "last" if checkpoint_path.match("last.ckpt") else "best"
    representations_dir = Path(experiment_dir / "representations/")
    representations_dir.mkdir(parents=True, exist_ok=True)

    output = dict()
    for idx in range(z.shape[1]):
        output[f"z-{idx}"] = z[:, idx]
    output["y"] = y
    output["s"] = s

    df = pd.DataFrame(output)
    df.to_csv(representations_dir / f"{ckpt_type}.csv", index=False)


class BaseDB(object):
    def __init__(self, filepath):
        self.db = TinyDB(filepath)
        self.metadata = self.db.table("metadata", cache_size=30)
        self.results = self.db.table("results", cache_size=30)
        self._query = Query()
        self.empty_cols = [
            "experiment_uuid",
            "replicate_id",
            "measure_id",
            "result_id",
            "evaluator",
            "name",
            "estimator",
            "fold",
            "metric",
            "value",
        ]

    def _records2table(self, metadata: Records, results: Records) -> pd.DataFrame:
        """Get records and join tables"""
        if len(metadata) > 0 and len(results) > 0:
            metadata = pd.DataFrame.from_records(metadata)
            results = pd.DataFrame.from_records(results)

            # Expand metrics values and creates result_id
            records = [
                {"result_id": idx, "metric": k, "value": v}
                for idx, m in enumerate(results.pop("metrics"))
                for k, v in m.items()
            ]
            metrics = pd.DataFrame.from_records(records, index="result_id")
            results = (
                results.join(metrics)
                .reset_index(drop=False)
                .rename(columns=dict(index="result_id"))
            )
            return metadata.merge(results, on="experiment_uuid")
        else:
            return pd.DataFrame(None, columns=self.empty_cols)

    def query(self, cond: QueryInstance = None):
        metadata = self.metadata.all() if cond is None else self.metadata.search(cond)
        experiment_uuids = [md["experiment_uuid"] for md in metadata]
        cond = self._query.experiment_uuid.one_of(experiment_uuids)
        results = self.results.search(cond)
        return metadata, results

    def all(self):
        metadata = self.metadata.all()
        results = self.results.all()
        return self._records2table(metadata, results)

    def aggregate(self, df: pd.DataFrame, key="measure_id"):
        """
        Aggregate evaluation over replicates and evaluation folds.

        evaluation_id = replicate_id + ["evaluator", "name", "estimator"]

        Args:
            key: 'evaluation_uuid', 'evaluation_id' (default)
        """
        grouped = df.groupby([key, "metric"])["value"]
        df = df.assign(
            n=grouped.transform(len),
            mean=grouped.transform(np.mean),
            median=grouped.transform(np.mean),
            std=grouped.transform(np.std),
        )
        df.drop(["fold", "value", "result_id"], axis=1, inplace=True)
        return df.groupby([key, "metric"]).first()

    def where_model(self, model: str, latent_dim: int = None):
        if latent_dim is None:
            cond = self._query.model == model.upper()
        else:
            cond = (self._query.model == model.upper()) & (
                self._query.latent_dim == latent_dim
            )
        metadata = self.metadata.search(cond)
        experiment_uuids = [md["experiment_uuid"] for md in metadata]
        cond = self._query.experiment_uuid.one_of(experiment_uuids)
        results = self.results.search(cond)
        return self._records2table(metadata, results)

    def __repr__(self):
        return repr(self.db)

    def __del__(self):
        self.db.close()


class BaselineDB(BaseDB):
    def __init__(self, filepath):
        super().__init__(filepath)

    def where_baseline(self, aggregate=False, *args, **kwargs):
        table = self.all()
        table["model"] = "Baseline"
        return table if not aggregate else self.aggregate(table, *args, **kwargs)

    def get_tables(self):
        return self.where_baseline()


class AlphaBetaTradeoffDB(BaseDB):
    def __init__(self, filepath):
        super().__init__(filepath)

    def where_cpfsi(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "CPFSI")
            &
            # (self._query.alpha >= 1) &
            (self._query.alpha > 0)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    def where_ibsi(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "IBSI")
            & (self._query.alpha > 0)
            & (self._query.alpha <= 1)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    def where_cpf(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "CPFSI")
            & (self._query.alpha >= 1)
            & (self._query.beta == 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        table["model"] = "CPF"
        return self.aggregate(table, key=key) if aggregate else table

    def where_cfb(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "CPFSI")
            & (self._query.alpha == 0)
            & (self._query.beta >= 1)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        table["model"] = "CFB"
        return self.aggregate(table, key=key) if aggregate else table

    def get_tables(self, aggregate=False, latent_dim=None):
        cpf = self.where_cpf(latent_dim=latent_dim, aggregate=aggregate)
        cfb = self.where_cfb(latent_dim=latent_dim, aggregate=aggregate)
        ibsi = self.where_ibsi(latent_dim=latent_dim, aggregate=aggregate)
        ours = self.where_cpfsi(latent_dim=latent_dim, aggregate=aggregate)
        return cpf, cfb, ibsi, ours


class BetaTradeoffDB(BaseDB):
    def __init__(self, filepath):
        super().__init__(filepath)

    def where_ibsi(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "IBSI")
            & (self._query.alpha > 0)
            & (self._query.alpha <= 1)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    def where_cfb(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "CPFSI")
            & (self._query.alpha == 0)
            & (self._query.beta >= 1)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        table["model"] = "CFB"
        return self.aggregate(table, key=key) if aggregate else table

    def where_cpfsi(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "CPFSI")
            &
            # (self._query.alpha >= 1) &
            (self._query.alpha > 0)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    def get_tables(self, latent_dim=None, aggregate=True):
        ibsi = self.where_ibsi(latent_dim=latent_dim, aggregate=aggregate)
        cfb = self.where_cfb(latent_dim=latent_dim, aggregate=aggregate)
        ours = self.where_cpfsi(latent_dim=latent_dim, aggregate=aggregate)
        return ibsi, cfb, ours


class SSLTradeoffDB(BaseDB):
    def __init__(self, filepath):
        super().__init__(filepath)

    # Unsupervised
    def where_vfae(self, latent_dim=None, aggregate=False, key="measure_id"):
        cond = self._query.model == "VFAE"
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        table["alpha"] = 1
        table["beta"] = 0
        return self.aggregate(table, key=key) if aggregate else table

    def where_cpf(self, latent_dim=None, aggregate=False, key="measure_id"):
        cond = (
            (self._query.model == "CPFSI")
            & (self._query.alpha > 0)
            & (self._query.beta == 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        table["model"] = "CPF"
        return self.aggregate(table, key=key) if aggregate else table

    # Fully Supervised
    def where_ibsi(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "IBSI")
            & (self._query.alpha > 0)
            & (self._query.alpha <= 1)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    def where_cpfsi(self, latent_dim=None, aggregate=False, key="measure_id"):
        cond = (
            (self._query.model == "CPFSI")
            & (self._query.alpha > 0)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    # Semi Supervised
    def where_semivfae(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = self._query.model == "SemiVFAE"
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        table["alpha"] = 1
        return self.aggregate(table, key=key) if aggregate else table

    def where_semiibsi(self, aggregate=True, latent_dim=None, key="measure_id"):
        cond = (
            (self._query.model == "SemiIBSI")
            & (self._query.alpha > 0)
            & (self._query.alpha <= 1)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    def where_semicpfsi(self, latent_dim=None, aggregate=False, key="measure_id"):
        cond = (
            (self._query.model == "SemiCPFSI")
            & (self._query.alpha > 0)
            & (self._query.beta > 0)
        )
        if latent_dim is not None:
            cond = cond & (self._query.latent_dim == latent_dim)
        metadata, results = self.query(cond)
        table = self._records2table(metadata, results)
        return self.aggregate(table, key=key) if aggregate else table

    def get_tables(self, latent_dim=None, aggregate=True):
        kwargs = dict(latent_dim=latent_dim, aggregate=aggregate)
        vfae = self.where_vfae(**kwargs)
        cpf = self.where_cpf(**kwargs)
        ibsi = self.where_ibsi(**kwargs)
        cpfsi = self.where_cpfsi(**kwargs)
        semivfae = self.where_semivfae(**kwargs)
        semiibsi = self.where_semiibsi(**kwargs)
        semicpfsi = self.where_semicpfsi(**kwargs)
        return vfae, cpf, ibsi, cpfsi, semivfae, semiibsi, semicpfsi
