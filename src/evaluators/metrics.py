import torch
from torch import Tensor
import torchmetrics as tm
from torchmetrics import Metric


class Discrimination(Metric):
    full_state_update = True
    # full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.eps = 1e-10
        self.threshold = 0.5
        self.add_state("yhat_s0", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("yhat_s1", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_s0", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_s1", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, context: torch.Tensor):
        assert preds.shape == context.shape
        preds = (preds > self.threshold).int()
        s0, s1 = context == 0, context == 1
        self.yhat_s0 += torch.sum(preds[s0])
        self.yhat_s1 += torch.sum(preds[s1])
        self.total_s0 += torch.sum(s0)
        self.total_s1 += torch.sum(s1)

    def compute(self):
        ps0 = self.yhat_s0.float() / (self.total_s0 + self.eps)
        ps1 = self.yhat_s1.float() / (self.total_s1 + self.eps)
        return torch.abs(ps1 - ps0)


class EqualizedOdds(Metric):
    # full_state_update = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.eo = [Discrimination() for _ in range(2)]

    def update(self, preds: torch.Tensor, target: torch.Tensor, context: torch.Tensor):
        t0, t1 = target == 0, target == 1
        if torch.any(t0):
            self.eo[0].update(preds[t0], None, context[t0])
        if torch.any(t1):
            self.eo[1].update(preds[t1], None, context[t1])

    def compute(self):
        return torch.max(self.eo[0].compute(), self.eo[1].compute())


class ErrorGap(Metric):
    full_state_update = True
    # full_state_update = False

    def __init__(self):
        """
        Computer the accuracy gap for the case when y and s are binary
        """
        super().__init__()
        # num_classes = 2
        self.accuracies = [tm.Accuracy() for _ in range(2)]

    def update(self, preds: Tensor, target: Tensor, context: Tensor):
        assert preds.shape == target.shape == context.shape
        s0, s1 = context == 0, context == 1
        if torch.any(s0):
            self.accuracies[0].update(preds[s0], target[s0])
        if torch.any(s1):
            self.accuracies[1].update(preds[s1], target[s1])

    def compute(self):
        return torch.abs(self.accuracies[0].compute() - self.accuracies[1].compute())


class MetricByContext(Metric):
    full_state_update = False

    def __init__(self, num_groups: int, metric_cls: Metric, *args, **kwargs):
        super().__init__()
        self.num_groups = num_groups
        self.metrics = [metric_cls(*args, **kwargs) for _ in range(num_groups)]

    def update(self, preds: Tensor, target: Tensor, context: Tensor):
        assert preds.shape == target.shape == context.shape
        cvals = torch.unique(context)
        for cval in cvals:
            bool_idx = context == cval
            if torch.any(bool_idx):
                self.metrics[cval.int().item()].update(
                    preds[bool_idx], target[bool_idx]
                )

    def compute(self):
        return torch.tensor(
            [self.metrics[idx].compute() for idx in range(self.num_groups)]
        )


class MetricGap(Metric):
    full_state_update = False

    def __init__(self, metric_cls: Metric, *args, **kwargs):
        super().__init__()
        self.metrics = [metric_cls(*args, **kwargs) for _ in range(2)]

    def update(self, preds: Tensor, target: Tensor, context: torch.Tensor):
        assert preds.shape == target.shape == context.shape
        s0, s1 = context == 0, context == 1
        if torch.any(s0):
            self.metrics[0].update(preds[s0], target[s0])
        if torch.any(s1):
            self.metrics[1].update(preds[s1], target[s1])

    def compute(self):
        return torch.abs(self.metrics[0].compute() - self.metrics[1].compute())
