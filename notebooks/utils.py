import os
import sys

from typing import Sequence
from itertools import product

import numpy as np
import pandas as pd

import pygmo as pg
from pymoo.indicators.hv import HV
from scipy import stats
from scipy.spatial.distance import euclidean
from tinydb import Query, TinyDB
from tinydb.queries import QueryInstance

Records = Sequence[dict]


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


def expand_solutions(t1, t2):
    transposed = zip(t1, t2)
    return list(product(*transposed))


class hidden_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def remove_stratified_metrics(*dfs):
    grouped_metrics = ["accuracies", "precisions", "recalls"]
    return [df.loc[~df["metric"].isin(grouped_metrics)].copy() for df in dfs]


def stderr(x):
    return np.std(x) / np.sqrt(len(x))


def pivot_table(result):
    index = ["dataset", "model", "alpha", "beta", "seed"]
    columns = ["name", "metric"]
    values = "value"
    aggfunc = np.median

    table = result.pivot_table(
        index=index, columns=columns, values=values, aggfunc=aggfunc
    )
    return table


def get_solutions(table, property):
    reconstruction = ("0", "mae")
    if property == "discrimination":
        property = ("y", "discrimination")
    elif property == "leakage":
        property = ("s", "accuracy")
    else:
        raise ValueError()
    pred_error = ("y", "1-accuracy")
    table[("y", "1-accuracy")] = 1 - table[("y", "accuracy")]
    columns = [reconstruction, property, pred_error]
    return table[columns]


def covering_indicator(pf1, pf2):
    """C-metric: proportion of p2 dominated by p1

    Minimization is assumed

    Returns:
        - [0,1]: 1 if p1 dominates all p2, 0 if no p2 is dominated by p1
    """
    if isinstance(pf1, np.ndarray):
        pf1 = pf1.tolist()
    if isinstance(pf2, np.ndarray):
        pf2 = pf2.tolist()
    n2 = len(pf2)
    n_dominated = 0
    for p2 in pf2:
        for p1 in pf1:
            if p1 == p2 or pg.pareto_dominance(p1, p2):
                n_dominated += 1
                break
    return n_dominated / n2


def er(pf, pf_ref):
    """Number of non-dominated objective vectors which belong to the Pareto front."""
    if isinstance(pf, np.ndarray):
        pf = pf.tolist()
    if isinstance(pf_ref, np.ndarray):
        pf_ref = pf_ref.tolist()
    n = len(pf)
    counter = 0
    for p in pf:
        counter += int(p not in pf_ref)
    return counter / n


def c1r(pf, pf_ref):
    """Number of non-dominated objective vectors which belong to the Pareto front.

    Ratio of the reference points

    """
    if isinstance(pf, np.ndarray):
        pf = pf.tolist()
    if isinstance(pf_ref, np.ndarray):
        pf_ref = pf_ref.tolist()
    n_ref = len(pf_ref)
    counter = 0
    for p in pf:
        counter += int(p in pf_ref)
    return counter / n_ref


def c2r(pf, pf_ref):
    """Ratio of non-dominated points by the reference set"""
    if isinstance(pf, np.ndarray):
        pf = pf.tolist()
    if isinstance(pf_ref, np.ndarray):
        pf_ref = pf_ref.tolist()
    n = len(pf)
    n_dominant = 0
    for p in pf:
        exists = True
        for p_ref in pf_ref:
            if pg.pareto_dominance(p_ref, p):
                exists = False
                break
        n_dominant += int(exists)
    return n_dominant / n


def d_metric(pf1, pf2, ref_point):
    """D-Metric"""
    if isinstance(pf1, np.ndarray):
        pf1 = pf1.tolist()
    if isinstance(pf2, np.ndarray):
        pf2 = pf2.tolist()
    hv = HV(ref_point=ref_point)
    return hv(np.array(pf1 + pf2)) - hv(np.array(pf2))


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


class BaseTable(object):
    def __init__(self, results_filepaths):
        self.results_filepaths = results_filepaths

    def print_empty(self, *dbs):
        for idx, df in enumerate(dbs):
            if df.empty:
                print(idx, df)

    def get_tables(self):
        pass

    def _add_metric_target(self, *dfs):
        for df in dfs:
            df["metric_target"] = df["metric"].str.cat(df["name"], sep="_")
        return dfs

    def _filter_seed(self, seed, *dfs):
        return [df.loc[df["seed"] == seed] for df in dfs]

    def _remove_stratified_metrics(self, *dfs):
        grouped_metrics = ["accuracies", "precisions", "recalls"]
        return [df.loc[~df["metric"].isin(grouped_metrics)].copy() for df in dfs]

    def _pivot_tables_args(self, base_index, aggregation):
        if "estimator" in aggregation:
            raise ValueError

        if aggregation is None or aggregation == "none":
            # No aggregation
            index = base_index + ["experiment_id", "replicate_id", "measure_id"]
        elif aggregation == "measure":
            # Aggregate by measure
            index = base_index + ["experiment_id", "replicate_id"]
        elif aggregation == "replicate":
            # Aggregate by replicate
            index = base_index + ["experiment_id", "measure_id"]
        elif aggregation == "all" or aggregation == "both":
            index = base_index
        else:
            index = aggregation if isinstance(aggregation, list) else [aggregation]

        values = "value"  # "mean", "median"
        columns = list()
        if "estimator" not in base_index:
            columns += ["estimator"]
        columns += ["name"]
        if "metric" not in base_index:
            columns += ["metric"]
        # print(columns)

        aggfunc = np.median
        # aggfunc = [np.mean, stats.sem]

        index += ["model", "alpha", "beta"]
        index = list(set(index))  # Remove duplicates

        # print(values, index, columns, aggfunc)
        return values, index, columns, aggfunc


class AlphaBetaTradeoffTable(BaseTable):
    def __init__(
        self,
        results_filepaths,
        prediction="accuracy",
        fairness="discrimination",
        reconstruction="mae",
    ):
        """
        Args:
            estimator: "dummy", "lr", "rf"
            prediction: "accuracy", "precision", "recall", "auc"
            reconstruction: "mse", "rmse", "mae",
            fairness: "discrimination", "equalized_odds", "error_gap"
        """
        super().__init__(results_filepaths)
        self.prediction = prediction
        self.fairness = fairness
        self.reconstruction = reconstruction

    def get_tables(self, latent_dim=None, aggregate=False):
        cpf, cfb, ibsi, ours = list(), list(), list(), list()
        for results_filepath in self.results_filepaths:
            ab = AlphaBetaTradeoffDB(results_filepath)
            cpf_, cfb_, ibsi_, ours_ = ab.get_tables(
                latent_dim=latent_dim, aggregate=aggregate
            )
            cpf.append(cpf_)
            cfb.append(cfb_)
            ibsi.append(ibsi_)
            ours.append(ours_)
        cpf = pd.concat(cpf)
        cfb = pd.concat(cfb)
        ibsi = pd.concat(ibsi)
        ours = pd.concat(ours)
        return cpf, cfb, ibsi, ours

    def prepare_tables(
        self,
        base_index=None,
        aggregation="none",
        seed=None,
        latent_dim=None,
    ):
        if base_index is None:  # Features used in FaceGrid
            base_index = list()

        cpf, cfb, ibsi, ours = self.get_tables(latent_dim=latent_dim, aggregate=False)

        if seed:
            cpf, cfb, ibsi, ours = self._filter_seed(seed, cpf, cfb, ibsi, ours)

        # Remove stratified metrics
        cpf, cfb, ibsi, ours = self._remove_stratified_metrics(cpf, cfb, ibsi, ours)
        values, index, columns, aggfunc = self._pivot_tables_args(
            base_index, aggregation
        )

        if not cpf.empty:
            cpf = cpf.pivot_table(
                index=index, columns=columns, values=values, aggfunc=aggfunc
            )

        if not cfb.empty:
            cfb = cfb.pivot_table(
                index=index, columns=columns, values=values, aggfunc=aggfunc
            )

        if not ours.empty:
            ours = ours.pivot_table(
                index=index, columns=columns, values=values, aggfunc=aggfunc
            )

        return cpf, cfb, ibsi, ours
