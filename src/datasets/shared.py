from typing import List
from dataclasses import dataclass
import random

import numpy as np

from torch.utils.data import Dataset, Subset, SubsetRandomSampler

import sklearn as sk
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR

from ..evaluators import (
    EvaluationCommittee,
    baseline,
    baseline,
    representation_prediction_fairness,
    representation_reconstruction,
    representation_leakage,
    model_prediction_fairness,
)
from .. import utils


BIN = "bin"
ID = "id"
NUM = "num"
ORD = "ord"
CAT = "cat"


@dataclass
class F(object):
    type: str
    _categories: list = None
    missing: int = 0

    @property
    def categories(self):
        return list(map(str, self._categories))


def split_train_val(dataset, val_prop=0.2):
    assert 0 < val_prop < 1
    num_samples = len(dataset)
    idxs = list(range(num_samples))
    np.random.shuffle(idxs)
    train_end = int((1 - val_prop) * num_samples)
    train_idxs = idxs[:train_end]
    val_idxs = idxs[train_end:]
    assert len(train_idxs) + len(val_idxs) == num_samples, "the split is not valid"
    return train_idxs, val_idxs


def subset_per_class(dataset, num_samples_per_class=100, num_classes=10):
    """Get a predefined number of sample indexes per class

    Args:
        dataset (Dataset)
        num_samples_per_class (int):
        num_classes (int):

    Returns:
        List[List[int]]: Sample indexes for each class
        List[int]: Remaining sample indexes

    """
    idxs = [idx for idx in range(len(dataset))]
    random.shuffle(idxs)
    remaining_idxs = list()
    class_idxs = [list() for _ in range(num_classes)]
    targets = dataset.dataset.targets
    for idx in idxs:
        label = targets[idx].int()
        if len(class_idxs[label]) < num_samples_per_class:
            class_idxs[label].append(idx)
        else:
            remaining_idxs.append(idx)
    return utils.flatten(class_idxs), utils.flatten(remaining_idxs)


def subset_per_class(dataset, num_samples_per_class=100, num_classes=10):
    """Get a predefined number of sample indexes per class

    Args:
        dataset (Dataset)
        num_samples_per_class (int):
        num_classes (int):

    Returns:
        List[List[int]]: Sample indexes for each class
        List[int]: Remaining sample indexes

    """
    idxs = [idx for idx in range(len(dataset))]
    random.shuffle(idxs)
    remaining_idxs = list()
    class_idxs = [list() for _ in range(num_classes)]
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    targets = dataset.targets if hasattr(dataset, "targets") else None
    for idx in idxs:
        label = dataset[idx][1].int() if targets is None else targets[idx].int()
        if len(class_idxs[label]) < num_samples_per_class:
            class_idxs[label].append(idx)
        else:
            remaining_idxs.append(idx)
    return utils.flatten(class_idxs), utils.flatten(remaining_idxs)


def subset_per_class_per_group(
    dataset, num_samples_per_class=100, num_classes=10, num_groups=2
):
    """Get a label per class per group"""
    assert num_samples_per_class >= num_groups, "num_samples_per_class < num_groups"
    idxs = [idx for idx in range(len(dataset))]
    random.shuffle(idxs)
    remaining_idxs = list()
    class_idxs = [
        [list() for _ in range(num_groups)] for _ in range(num_classes)
    ]  # FIXME
    for idx in idxs:
        label = (
            dataset.labels[idx].int()
            if hasattr(dataset, "labels")
            else dataset[idx][1].int()
        )
        group = (
            dataset.context[idx].int()
            if hasattr(dataset, "context")
            else dataset[idx][2].int()
        )
        # FIXME: ...
        if len(class_idxs[label][group]) < num_samples_per_class // num_groups:  # FIXME
            class_idxs[label][group].append(idx)  # FIXME
        else:
            remaining_idxs.append(idx)
    return utils.flatten(class_idxs), utils.flatten(remaining_idxs)


def get_semisup_samplers(dataset, num_samples_per_class, num_classes):
    labeled_idxs, unlabeled_idxs = subset_per_class(
        dataset, num_samples_per_class=num_samples_per_class, num_classes=num_classes
    )
    return SubsetRandomSampler(labeled_idxs), SubsetRandomSampler(unlabeled_idxs)


class TabularCommittee(EvaluationCommittee):
    def __init__(self, data_dict, recon_idx=0):
        """
        Base EvaluationCommittee
        -> What to evaluate, skEstimator configurations, etc.
        """
        self.recon_idx = recon_idx
        self.data_dict = data_dict  # Map[target, target_(name, dim, type)]
        self.logist_regression_config = dict(max_iter=400)  # Increase max of iter
        self.random_forest_classifier_config = dict(n_estimators=100)  # Default
        self.random_forest_regressor_config = dict(n_estimators=100, max_depth=8)

        self.baseline_target_estimators = self.init_baseline_estimators()
        self.prediction_fairness_estimators = self.init_prediction_fairness_estimators()
        self.leakage_estimators = self.init_leakage_estimators()
        self.reconstruction_estimators = self.init_reconstruction_estimators()

        self.logger = utils.get_logger(__name__)

    def init_baseline_estimators(self):
        estimators = [
            ("y", DummyClassifier(strategy="prior")),
            ("y", LogisticRegression(**self.logist_regression_config)),
            ("y", RandomForestClassifier(**self.random_forest_classifier_config)),
        ]
        return estimators

    def init_prediction_fairness_estimators(self):
        estimators = [
            ("y", LogisticRegression(**self.logist_regression_config)),
            ("y", RandomForestClassifier(**self.random_forest_classifier_config)),
        ]
        return estimators

    def init_leakage_estimators(self):
        estimators = [
            ("s", LogisticRegression(**self.logist_regression_config)),
            ("s", RandomForestClassifier(**self.random_forest_classifier_config)),
        ]
        return estimators

    def init_reconstruction_estimators(self):
        recon_idx = self.recon_idx
        if self.data_dict[recon_idx][2] == "num":
            estimators = [
                (recon_idx, LinearRegression()),
                (
                    recon_idx,
                    RandomForestRegressor(**self.random_forest_regressor_config),
                ),
            ]
        elif self.data_dict[recon_idx][2] == "bin":
            self.reconstruction_estimators = [
                (recon_idx, LogisticRegression(**self.logist_regression_config)),
                (
                    recon_idx,
                    RandomForestRegressor(**self.random_forest_regressor_config),
                ),
            ]
        else:
            raise NotImplementedError
        return estimators

    def baseline(self, test_dataloader=None):
        """Representation Baseline"""
        import ray

        self.logger.info("Running baseline")

        if not ray.is_initialized():
            ray.init(num_cpus=2, local_mode=True)

        @ray.remote
        def _representation_baseline(*args, **kwargs):
            return baseline(*args, **kwargs)

        results = list()
        for _, estimator in self.baseline_target_estimators:
            target, context = "y", "s"
            _, target_dim, target_type = self.data_dict[target]
            _, context_dim, context_type = self.data_dict[context]
            out = _representation_baseline.remote(
                target_name=target,
                target_dim=target_dim,
                target_type=target_type,
                context_name=context,
                context_dim=context_dim,
                context_type=context_type,
                test_dataloader=test_dataloader,
                sk_estimator=sk.base.clone(estimator),
            )
            results.append(out)
        return ray.get(results)

    def prediction_baseline(self, train_dataloader, test_dataloader):
        """Prediction Baseline"""
        import ray

        self.logger.info("Running baseline")

        if not ray.is_initialized():
            ray.init(num_cpus=2, local_mode=True)

        @ray.remote
        def _prediction_baseline(*args, **kwargs):
            return prediction_baseline(*args, **kwargs)

        results = list()
        for _, estimator in self.baseline_target_estimators:
            target, context = "y", "s"
            _, target_dim, target_type = self.data_dict[target]
            _, context_dim, context_type = self.data_dict[context]
            out = _prediction_baseline.remote(
                target_name=target,
                target_dim=target_dim,
                target_type=target_type,
                context_name=context,
                context_dim=context_dim,
                context_type=context_type,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                sk_estimator=sk.base.clone(estimator),
            )
            results.append(out)
        return ray.get(results)

    def evaluate(self, model, dataloader=None):
        """Evaluate representations"""
        results = list()

        # Predicton and Fairness
        # Representations
        for target, estimator in self.prediction_fairness_estimators:
            self.logger.info(f"Running estimator: {estimator.__class__} ")
            _, target_dim, target_type = self.data_dict[target]
            _, context_dim, _ = self.data_dict["s"]
            out = representation_prediction_fairness(
                target=target,
                target_dim=target_dim,
                target_type=target_type,
                context_dim=context_dim,
                model=model,
                dataloader=dataloader,
                sk_estimator=estimator,
            )
            results.append(out)

        # Model
        if hasattr(model, "predictor") or hasattr(model, "qy_z1"):
            self.logger.info(f"Running predictive posterior: {model.__class__} ")
            _, target_dim, target_type = self.data_dict[target]
            _, context_dim, _ = self.data_dict["s"]
            out = model_prediction_fairness(
                target=target,
                target_dim=target_dim,
                target_type=target_type,
                context_dim=context_dim,
                model=model,
                dataloader=dataloader,
            )
            results.append(out)

        # Reconstruction
        for input_idx, estimator in self.reconstruction_estimators:
            self.logger.info(f"Running estimator: {estimator.__class__} ")
            _, input_dim, input_type = self.data_dict[input_idx]
            _, context_dim, _ = self.data_dict["s"]
            out = representation_reconstruction(
                input_idx=input_idx,
                input_dim=input_dim,
                input_type=input_type,
                context_dim=context_dim,
                model=model,
                dataloader=dataloader,
                sk_estimator=estimator,
            )
            results.append(out)

        # Privacy
        for target, estimator in self.leakage_estimators:
            self.logger.info(f"Running estimator: {estimator.__class__} ")
            _, target_dim, target_type = self.data_dict["s"]
            out = representation_leakage(
                target_dim=target_dim,
                target_type=target_type,
                model=model,
                dataloader=dataloader,
                sk_estimator=estimator,
            )
            results.append(out)

        return results
