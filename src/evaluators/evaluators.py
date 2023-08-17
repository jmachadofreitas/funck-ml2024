from abc import ABCMeta, abstractmethod
from typing import Any, Union, Dict, List, Tuple, Type, Sequence

import pytorch_lightning
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm

import sklearn as sk
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from .metrics import *
from .shared import *


Number = Union[int, float]

SkEstimator = Union[
    DummyRegressor,
    DummyClassifier,
    LogisticRegression,
    LinearRegression,
    RandomForestClassifier,
    RandomForestRegressor,
    SVR,
    SVC,
]


def _assert_target(target_dim, target_type):
    assert target_type in ("bin", "cat", "num", "logit")
    if target_type == "bin":
        assert target_dim == 1
    elif target_type == "cat":
        assert target_dim > 1
    else:
        assert target_dim > 0


class Evaluator(tm.MetricCollection, metaclass=ABCMeta):
    datatypes = ("bin", "cat", "num", "logit", "mixed")
    transform = dict(
        bin=torch.sigmoid, cat=torch.softmax, num=None, logits=None, mixed=None
    )

    def __init__(
        self,
        evaluator_name: str,
        name: str,
        shape: Sequence[int],
        type: str,
        metrics,
        *additional_metrics,
        prefix=None,
        postfix=None,
    ):
        """
        Task specific Metric Collection compatible with PyTorch, PytorchLightning and ScikitLearn

        Encapsulate metrics collection creation
        - Evaluate a single model output provided as unormalized scores (logits)
        - Metrics on Evaluator must have the same update signature
        - Task specific - logits name, shape, and type

        Args:
             name (str): name of what we are evaluating

        References:
            https://torchmetrics.readthedocs.io/en/stable/pages/classification.html#input-types
        """
        # prefix = f"{evaluator_name}/"
        super().__init__(metrics, *additional_metrics, prefix=prefix, postfix=postfix)
        self.evaluator_name = evaluator_name
        self.name = name
        self.shape = shape
        self.type = type

    @staticmethod
    def _to_torch(ndarray):
        return (
            ndarray if isinstance(ndarray, torch.Tensor) else torch.from_numpy(ndarray)
        )

    def _convert_to_torch(self, batch):
        return [el if el is None else self._to_torch(el) for el in batch]

    @staticmethod
    def _sklearn_predict(model: SkEstimator, batch: Tuple) -> np.ndarray:
        pred = model.predict(batch[0])
        return pred

    @staticmethod
    def _torch_predict(
        model: Union[pl.LightningModule, nn.Module],
        batch,
        batch_idx=0,
        dataloader_idx=0,
    ) -> Dict[str, Any]:
        model.eval()
        return model.predict_step(batch, batch_idx, dataloader_idx)

    def _predict(
        self, model, batch, batch_idx, dataloader_idx=None
    ) -> Union[np.ndarray, Dict[str, Tensor]]:
        """
        Run inference for pl.LightningModule and scikit-learn

        Args:
            model: SkEstimator, LightningModule
            target_name: target names for when this is not provided by the model prediction output
        """

        if isinstance(model, sk.base.BaseEstimator):
            out = self._sklearn_predict(model, batch)
        elif isinstance(model, (nn.Module, pl.LightningModule)):
            out = self._torch_predict(model, batch)
        else:
            raise NotImplementedError
        return out

    def _pre_update(self, preds, target):
        """
        Args:
            preds: are logits if in torch.Tensor
            target:
        """
        fn = self.transform[self.type]
        if isinstance(preds, torch.Tensor) and fn:
            preds = fn(preds)
            target = self._to_torch(target)
        else:
            preds, target = self._to_torch(preds), self._to_torch(target)
        preds = preds.float() if self.type == "bin" else preds
        return preds, target

    def update(self, preds, target):
        preds, target = self._pre_update(preds, target)
        for _, m in self.items(keep_base=True):
            m.update(preds, target)

    def update_with_context(self, preds, target, context):
        preds, target = self._pre_update(preds, target)
        context = context if context is None else self._to_torch(context)
        for _, m in self.items(keep_base=True):
            m.update(preds, target, context)

    def compute(self) -> Dict[str, Number]:
        return {k: v.tolist() for k, v in super().compute().items()}

    @abstractmethod
    def evaluate(
        self,
        model: Union[SkEstimator, pl.LightningModule],
        dataloaders: Union[DataLoader, Tuple],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class EvaluationCommittee(metaclass=ABCMeta):
    """Base class for Dataset specific EvaluationCommittee"""

    @abstractmethod
    def baseline(self, test_dataloader):
        raise NotImplementedError

    def evaluate(self, model, dataloader=None):
        raise NotImplementedError


class PredictionEvaluator(Evaluator):
    evaluator_name = "prediction"

    def __init__(self, target_name, target_dim, target_type):
        """Collection of Prediction Metrics"""
        if target_type in ("bin", "cat"):
            num_classes = target_dim + 1
            if num_classes > 2:
                # TODO: correct?
                kwargs = dict(num_classes=num_classes, multiclass=True)
            else:
                kwargs = dict()
            metrics = OrderedDict(
                accuracy=tm.Accuracy(**kwargs),
                precision=tm.Precision(**kwargs),
                recall=tm.Recall(**kwargs),
                f1score=tm.F1Score(**kwargs),
                # auc=tm.AUROC(num_classes=num_classes),
                # confmat=tm.ConfusionMatrix(num_classes=num_classes)
            )
        elif target_type == "num":
            metrics = OrderedDict(
                rmse=tm.MeanSquaredError(squared=False),
                mae=tm.MeanAbsoluteError(),
                mape=tm.MeanAbsolutePercentageError(),
            )
        else:
            raise NotImplementedError
        super().__init__(
            self.evaluator_name,
            name=target_name,
            shape=(target_dim,),
            type=target_type,
            metrics=metrics,
        )

    def evaluate(self, model, dataloader: Union[DataLoader, List[Tuple]]):
        for batch_idx, batch in enumerate(dataloader):
            x, y, s = batch
            out = self._predict(model, batch, batch_idx)
            y_hat = out["y"] if isinstance(out, dict) else out  # Unpack
            self.update(y_hat, y)

        results = OrderedDict(
            evaluator=self.evaluator_name,
            name=self.name,
            dim=self.shape[0],
            type=self.type,
            metrics=self.compute(),
        )
        return results


class FairnessEvaluator(Evaluator):
    evaluator_name = "fairness"

    def __init__(self, target_name, target_dim, target_type, context_dim, context_type):
        """Collection of Fairness Metrics"""
        context_name = "s"
        if target_type == "bin" and context_type == "bin":
            metrics = OrderedDict(
                discrimination=Discrimination(),
                equalized_odds=EqualizedOdds(),
                error_gap=ErrorGap(),
                accuracies=MetricByContext(2, tm.Accuracy),
                recalls=MetricByContext(2, tm.Recall),
                precisions=MetricByContext(2, tm.Precision),
                # aucs=MetricByContext(2, tm.AUROC),
            )
        elif target_type == "num" and context_type == "bin":
            metrics = OrderedDict(
                rmse_gap=MetricGap(tm.MeanSquaredError, squared=False),
                mae_gap=MetricGap(tm.MeanAbsoluteError),
                mape_gap=MetricGap(tm.MeanAbsolutePercentageError),
            )
        else:
            raise NotImplementedError
        super().__init__(
            self.evaluator_name,
            name=target_name,
            shape=(target_dim,),
            type=target_type,
            metrics=metrics,
        )
        self.context_name = context_name
        self.context_dim = context_dim
        self.context_type = context_type

    def update(self, preds: torch.Tensor, target: torch.Tensor, context: torch.Tensor):
        self.update_with_context(preds, target, context)

    def evaluate(self, model, dataloader, true_dataloader=None):
        if true_dataloader is None:
            for batch_idx, batch in enumerate(dataloader):
                _, y, s = batch
                out = self._predict(model, batch, batch_idx)
                y_hat = out["y"] if isinstance(out, dict) else out  # Unpack
                self.update(y_hat, y, s)
        else:
            for batch_idx, (batch_fake, batch_true) in enumerate(
                zip(dataloader, true_dataloader)
            ):
                _, _, s_fake = batch_fake
                _, y, s_true = batch_true
                out = self._predict(model, batch_fake, batch_idx)
                y_hat = out["y"] if isinstance(out, dict) else out  # Unpack
                self.update(y_hat, y, s_true)

        results = OrderedDict(
            evaluator=self.evaluator_name,
            name=self.name,
            dim=self.shape[0],
            type=self.type,
            metrics=self.compute(),
        )
        return results


class PrivacyEvaluator(Evaluator):
    evaluator_name = "privacy"

    def __init__(self, target_dim, target_type):
        """Collection of Privacy/Leakage Metrics"""
        target_name = "s"
        if target_type in ("bin", "cat"):
            num_classes = target_dim + 1
            if num_classes > 2:
                # TODO: correct?
                kwargs = dict(num_classes=num_classes, multiclass=True)
            else:
                kwargs = dict()
            metrics = OrderedDict(
                accuracy=tm.Accuracy(**kwargs),
                precision=tm.Precision(**kwargs),
                recall=tm.Recall(**kwargs),
                f1score=tm.F1Score(**kwargs),
            )
        elif target_type == "num":
            metrics = OrderedDict(
                rmse=tm.MeanSquaredError(squared=False),
                mae=tm.MeanAbsoluteError(),
                mape=tm.MeanAbsolutePercentageError(),
            )
        else:
            raise NotImplementedError
        super().__init__(
            self.evaluator_name,
            name=target_name,
            shape=(target_dim,),
            type=target_type,
            metrics=metrics,
        )

    def evaluate(self, model, dataloader):
        for batch_idx, batch in enumerate(dataloader):
            x, y, s = batch
            out = self._predict(model, batch, batch_idx)
            s_hat = out["s"] if isinstance(out, dict) else out  # Unpack
            self.update(s_hat, s)

        results = OrderedDict(
            evaluator=self.evaluator_name,
            name=self.name,
            dim=self.shape[0],
            type=self.type,
            metrics=self.compute(),
        )
        return results


class ReconstructionEvaluator(Evaluator):
    """Feature Reconstruction"""

    evaluator_name = "reconstruction"

    def __init__(self, input_idx, input_name, input_dim, input_type):
        """Collection of Prediction Metrics for a single reconstructed feature"""

        if input_type in ("bin", "cat"):
            num_classes = target_dim + 1
            if num_classes > 2:  # TODO: correct?
                kwargs = dict(num_classes=num_classes, multiclass=True)
            else:
                kwargs = dict()
            metrics = OrderedDict(
                accuracy=tm.Accuracy(**kwargs),
                precision=tm.Precision(**kwargs),
                recall=tm.Recall(**kwargs),
                f1score=tm.F1Score(**kwargs),
            )
        elif input_type == "num":
            metrics = OrderedDict(
                rmse=tm.MeanSquaredError(squared=False),
                mae=tm.MeanAbsoluteError(),
                mape=tm.MeanAbsolutePercentageError(),
            )
        else:
            raise NotImplementedError

        super().__init__(
            self.evaluator_name,
            name=input_name,
            shape=(input_dim,),
            type=input_type,
            metrics=metrics,
        )
        self.input_idx = input_idx

    @staticmethod
    def pick_target(x, idx):
        if isinstance(idx, int):
            return x[:, idx]
        elif isinstance(idx, tuple) and len(idx) == 2:
            start, end = idx
            return x[:, start:end]  # One-hot -> labels ?
        else:
            raise ValueError(f"Unknow idxs='{idx}'")

    def evaluate(self, model, dataloader: Union[DataLoader, List[Tuple]]):
        for batch_idx, batch in enumerate(dataloader):
            x, t, s = batch
            out = self._predict(model, batch, batch_idx)
            t_hat = (
                self.pick_target(out["x"], self.input_idx)
                if isinstance(out, dict)
                else out
            )
            self.update(t_hat, t)

        results = OrderedDict(
            evaluator=self.evaluator_name,
            name=self.name,
            dim=self.shape[0],
            type=self.type,
            metrics=self.compute(),
        )
        return results


class ImageGenerationEvaluator(Evaluator):
    pass


def get_sk_estimator_code(estimator: SkEstimator):
    if isinstance(estimator, (DummyRegressor, DummyClassifier)):
        return "dummy"
    elif isinstance(estimator, (LinearRegression, LogisticRegression)):
        return "lr"
    elif isinstance(estimator, (RandomForestRegressor, RandomForestClassifier)):
        return "rf"
    elif isinstance(estimator, (SVR, SVC)):
        return "svm"
    else:
        NotImplementedError(f"Unknown SkEstimator {repr(estimator)}")


# Evaluation helpers
def baseline(
    target_name: str,
    target_dim: int,
    target_type: str,
    context_name: str,
    context_dim: int,
    context_type: str,
    test_dataloader,
    sk_estimator,
    n_splits=5,
    n_repeats=1,
    seed=0,
):
    _sk_estimator: SkEstimator = sk.base.clone(sk_estimator)  # Clone estimator
    x, y, s = dataloader2numpy(test_dataloader)

    if target_type in ("bin", "cat"):
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (x, y)
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (x,)

    prediction_eval = PredictionEvaluator(
        target_name=target_name, target_dim=target_dim, target_type=target_type
    )

    fairness_eval = FairnessEvaluator(
        target_name=target_name,
        target_dim=target_dim,
        target_type=target_type,
        context_dim=context_dim,
        context_type=context_type,
    )

    leakage_eval = PrivacyEvaluator(target_dim=target_dim, target_type=target_type)

    results = list()
    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(*args)):
        x_tr, x_te = x[tr_idx], x[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        s_tr, s_te = s[tr_idx], s[te_idx]
        estimator_code = get_sk_estimator_code(sk_estimator)
        estimator_repr = repr(sk_estimator)

        # Prediction and Fairness
        sk_estimator.fit(x_tr, y_tr)

        # Prediction
        out = prediction_eval.evaluate(sk_estimator, [(x_te, y_te, s_te)])
        out["estimator"] = estimator_code
        out["estimator_repr"] = estimator_repr
        out["fold"] = fold_idx
        results.append(out)
        prediction_eval.reset()

        # Fairness
        out = fairness_eval.evaluate(sk_estimator, [(x_te, y_te, s_te)])
        out["estimator"] = estimator_code
        out["estimator_repr"] = estimator_repr
        out["fold"] = fold_idx
        results.append(out)
        fairness_eval.reset()

        # Privacy
        _sk_estimator.fit(x_tr, s_tr)
        out = leakage_eval.evaluate(_sk_estimator, [(x_te, y_te, s_te)])
        out["estimator"] = estimator_code
        out["estimator_repr"] = estimator_repr
        out["fold"] = fold_idx
        results.append(out)
        leakage_eval.reset()
    return results


def prediction_baseline(
    target_name: str,
    target_dim: int,
    target_type: str,
    context_name: str,
    context_dim: int,
    context_type: str,
    train_dataloader,
    test_dataloader,
    sk_estimator,
):
    _sk_estimator: SkEstimator = sk.base.clone(sk_estimator)  # Clone estimator
    x_tr, y_tr, s_tr = dataloader2numpy(train_dataloader)
    x_te, y_te, s_te = dataloader2numpy(test_dataloader)

    results = list()
    prediction_eval = PredictionEvaluator(
        target_name=target_name, target_dim=target_dim, target_type=target_type
    )
    fairness_eval = FairnessEvaluator(
        target_name=target_name,
        target_dim=target_dim,
        target_type=target_type,
        context_dim=context_dim,
        context_type=context_type,
    )
    leakage_eval = PrivacyEvaluator(target_dim=target_dim, target_type=target_type)

    estimator_code = get_sk_estimator_code(sk_estimator)
    estimator_repr = repr(sk_estimator)

    # Prediction and Fairness
    sk_estimator.fit(x_tr, y_tr)

    # Prediction
    out = prediction_eval.evaluate(sk_estimator, [(x_te, y_te, s_te)])
    out["estimator"] = estimator_code
    out["estimator_repr"] = estimator_repr
    results.append(out)
    prediction_eval.reset()

    # Fairness
    out = fairness_eval.evaluate(sk_estimator, [(x_te, y_te, s_te)])
    out["estimator"] = estimator_code
    out["estimator_repr"] = estimator_repr
    results.append(out)
    fairness_eval.reset()

    # Privacy
    _sk_estimator.fit(x_tr, s_tr)
    out = leakage_eval.evaluate(_sk_estimator, [(x_te, y_te, s_te)])
    out["estimator"] = estimator_code
    out["estimator_repr"] = estimator_repr
    results.append(out)
    leakage_eval.reset()

    return results


def representation_prediction_fairness(
    target: str,
    target_dim: int,
    target_type: str,
    context_dim: int,
    model,
    dataloader,
    sk_estimator,
    n_splits=5,
    n_repeats=1,
    seed=0,
):
    target_name = target  # "y", <target_name>
    context_type = "bin"

    z, (x, y, s) = get_representations_and_numpy_dataset(model, dataloader)

    prediction_eval = PredictionEvaluator(
        target_name=target_name, target_dim=target_dim, target_type=target_type
    )

    fairness_eval = FairnessEvaluator(
        target_name=target_name,
        target_dim=target_dim,
        target_type=target_type,
        context_dim=context_dim,
        context_type=context_type,
    )

    if target_type in ("bin", "cat"):
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (z, y)
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (z,)

    results = list()
    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(*args)):
        z_tr, z_te = z[tr_idx], z[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        s_tr, s_te = s[tr_idx], s[te_idx]

        estimator_code = get_sk_estimator_code(sk_estimator)
        estimator_repr = repr(sk_estimator)
        sk_estimator.fit(z_tr, y_tr)

        # Prediction
        out = prediction_eval.evaluate(sk_estimator, [(z_te, y_te, s_te)])
        out["estimator"] = estimator_code
        out["estimator_repr"] = estimator_repr
        out["fold"] = fold_idx
        results.append(out)
        prediction_eval.reset()

        # Fairness
        out = fairness_eval.evaluate(sk_estimator, [(z_te, y_te, s_te)])
        out["estimator"] = estimator_code
        out["estimator_repr"] = estimator_repr
        out["fold"] = fold_idx
        results.append(out)
        fairness_eval.reset()

    return results


def get_model_predictor_code(model, s="s"):
    from ..models import IBSI, SemiIBSI, VFAE, SemiVFAE, CPFSI, SemiCPFSI

    if isinstance(model, (IBSI, SemiIBSI)):
        return "q(y|z)"
    elif isinstance(model, (VFAE, SemiVFAE)):
        return "q(y|z1)"
    elif isinstance(model, (CPFSI, SemiCPFSI)):
        return f"q(y|z,{s})"
    else:
        NotImplementedError(f"Unknown Model {repr(model)}")


def model_prediction_fairness(
    target: str,
    target_dim: int,
    target_type: str,
    context_dim: int,
    model,
    dataloader,
):
    target_name = target  # "y", <target_name>
    context_type = "bin"

    z, (x, y, s) = get_representations_and_torch_dataset(model, dataloader)

    prediction_eval = PredictionEvaluator(
        target_name=target_name, target_dim=target_dim, target_type=target_type
    )

    fairness_eval = FairnessEvaluator(
        target_name=target_name,
        target_dim=target_dim,
        target_type=target_type,
        context_dim=context_dim,
        context_type=context_type,
    )

    results = list()
    s_half = 0.5 * torch.ones_like(s)
    s_rand = torch.bernoulli(s_half)
    s_flip = (~s.bool()).int()
    esses = {"s": s, "s=.5": s_half, "S": s_rand, "1-s": s_flip}
    for skey, sval in esses.items():
        # Prediction
        out = prediction_eval.evaluate(model, [(x, y, sval)])
        out["estimator"] = get_model_predictor_code(model, s=skey)
        # out["posterior"] = get_model_predictor_code(model, s=skey)
        out["skey"] = skey
        out["fold"] = 0
        results.append(out)
        prediction_eval.reset()

        # Fairness
        out = fairness_eval.evaluate(model, [(x, y, sval)], true_dataloader=[(x, y, s)])
        out["estimator"] = get_model_predictor_code(model, s=skey)
        out["skey"] = skey
        out["fold"] = 0
        results.append(out)
        fairness_eval.reset()

    return results


def representation_reconstruction(
    input_idx: Union[int, Tuple[int]],
    input_dim: int,
    input_type: str,
    context_dim: int,
    model,
    dataloader,
    sk_estimator,
    n_splits=5,
    n_repeats=1,
    seed=0,
):
    input_name = str(input_idx)
    context_type = "bin"

    z, (x, y, s) = get_representations_and_numpy_dataset(model, dataloader)

    # Reconstruction
    reconstruction_eval = ReconstructionEvaluator(
        input_idx=input_idx,
        input_name=input_name,
        input_dim=input_dim,
        input_type=input_type,
    )

    # Reconstruction Fairness
    # ...

    t = reconstruction_eval.pick_target(x, input_idx)  # x_i

    if input_type in ("bin", "cat"):
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (z, t)
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (z,)

    results = list()
    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(*args)):
        z_tr, z_te = z[tr_idx], z[te_idx]
        t_tr, t_te = t[tr_idx], t[te_idx]
        s_tr, s_te = s[tr_idx], s[te_idx]

        estimator_code = get_sk_estimator_code(sk_estimator)
        estimator_repr = repr(sk_estimator)
        sk_estimator.fit(z_tr, t_tr)

        # Reconstruction
        out = reconstruction_eval.evaluate(sk_estimator, [(z_te, t_te, s_te)])
        out["estimator"] = estimator_code
        out["estimator_repr"] = estimator_repr
        out["fold"] = fold_idx
        results.append(out)
        reconstruction_eval.reset()

    return results


def representation_leakage(
    target_dim: int,
    target_type: str,
    model,
    dataloader,
    sk_estimator,
    n_splits=4,
    n_repeats=1,
    seed=0,
):
    z, (x, y, s) = get_representations_and_numpy_dataset(model, dataloader)

    if target_type in ("bin", "cat"):
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (z, s)
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        args = (z,)

    results = list()
    leakage_eval = PrivacyEvaluator(target_dim=target_dim, target_type=target_type)
    for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(*args)):
        z_tr, z_te = z[tr_idx], z[te_idx]
        s_tr, s_te = s[tr_idx], s[te_idx]

        sk_estimator.fit(z_tr, s_tr)
        out = leakage_eval.evaluate(sk_estimator, [(z_te, None, s_te)])
        out["estimator"] = get_sk_estimator_code(sk_estimator)
        out["estimator_repr"] = repr(sk_estimator)
        out["fold"] = fold_idx
        results.append(out)
        leakage_eval.reset()

    return results
