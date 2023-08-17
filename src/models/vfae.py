from typing import Mapping
from collections import OrderedDict
from functools import reduce
from pprint import pprint

import torch
import torch.distributions as D
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from ..utils import DatasetConfig, ExperimentConfig
from ..modules import (
    KLDivergence,
    LogMVNDiffLoss,
    FastMMD,
    combine_features,
    make_activation_fn,
    freeze,
)
from .mixin import ModelMixin


class VFAE(ModelMixin, LightningModule):
    """
    Unsupervised version of the Variational Fair Autoencoder

    References:
        Louizos et al. ...

    """

    def __init__(self, dconfig: DatasetConfig, econfig: ExperimentConfig):
        LightningModule.__init__(self)
        ModelMixin.__init__(self, dconfig, econfig)

        self.save_hyperparameters(ignore=["reconstruction_loss", "prediction_loss"])

        mconfig = self.econfig.model
        self.context_dim = dconfig.context_dim
        self.latent_dim = mconfig.latent_dim
        # self.alpha = float(mconfig.alpha)
        self.gamma = float(mconfig.gamma)
        self.num_samples = mconfig.num_samples  # TODO
        self.lr = mconfig.lr
        self.activation_fn = make_activation_fn(mconfig.nonlinearity)

        # Modules
        self.encoder = self.init_prob_cond_encoder(
            input_shape=dconfig.input_shape,
            latent_dim=mconfig.latent_dim,
            context_dim=dconfig.context_dim,
            hidden_dims=mconfig.encoder.hidden_dims,
            nonlinearity=mconfig.nonlinearity,
        )
        self.decoder = self.init_cond_decoder(
            latent_dim=mconfig.latent_dim,
            context_dim=dconfig.context_dim,
            hidden_dims=mconfig.decoder.hidden_dims,
            output_shape=dconfig.input_shape,
            nonlinearity=mconfig.nonlinearity,
        )

        # Losses
        self.kl_loss = KLDivergence("mvn_diag", "mvn_std")
        self.reconstruction_loss = self.init_reconstruction_loss()
        self.mmd_loss = self.init_mmd_loss()

        # Metrics
        metrics = self.init_metrics()
        self.metrics = self.configure_metrics(metrics) if metrics else dict()

    def forward(self, x, c):
        """Generate representations"""
        return self._encode(x, c)[0]

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            sync_dist=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, c = batch
        z_mean, _, _ = self._encode(x, c)
        recon_logits = self._decode(z_mean, c)
        return {
            "z": z_mean,
            "x": recon_logits,
        }

    def configure_optimizers(self):
        lr = self.lr
        params_non_frozen = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.Adam(params_non_frozen, lr=lr)
        return opt

    def _encode(self, x, c):
        xc = combine_features(x, c)
        loc, logscale = self.encoder(xc)
        if self.training:  # Stochastic encoder
            sample = self.encoder.dsample(loc, logscale)
            return sample, loc, logscale
        else:  # Deterministic encoder
            return loc, loc, logscale

    def _decode(self, z, c):
        zc = combine_features(z, c)
        logits = self.decoder(zc)
        return logits

    def _train_step(self, batch):
        loss, logs = self._step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            sync_dist=False,
        )
        self.log_dict(self.train_metrics)
        return loss

    def _eval_step(self, batch, prefix):
        loss, logs = self._step(batch)
        self.log_dict(
            {f"{prefix}/{metric_name}": value for metric_name, value in logs.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return logs

    def _update_metrics(self, x_logits, x, ctx):
        stage = "train" if self.training else "val"
        for k, e in self.metrics[stage].items():
            if k == "reconstruction":
                e.update(x_logits, x)

    def _compute_metrics(self, stage):
        reconstruction_metrics = self.metrics[stage]["reconstruction"].compute()
        return reconstruction_metrics

    def _step(self, batch):
        x, target, ctx = batch
        logs = OrderedDict()

        z, mean, logvar = self._encode(x, ctx)
        recon_logits = self._decode(z, ctx)

        recon_loss = self.reconstruction_loss(recon_logits, x)

        mmd_loss = 0.0
        if self.mmd_loss is not None:
            mmd_loss = self.mmd_loss(x[ctx == 0, ...], x[ctx == 1, ...])

        kl_div = self.kl_loss(mean, logvar).mean()

        if isinstance(recon_loss, tuple):
            recon_loss, recon_logs = recon_loss
            logs.update(recon_logs)

        if isinstance(kl_div, tuple):
            kl_div, kl_logs = kl_idv
            logs.update(kl_logs)

        loss = recon_loss + kl_div + self.gamma * mmd_loss

        with torch.no_grad():
            stage = "train" if self.training else "val"
            if stage in self.metrics:
                self._update_metrics(recon_logits, x, ctx)
                reconstruction_metrics = self._compute_metrics(stage)

            logs = OrderedDict(
                loss=loss,
                recon_loss=recon_loss,
                kl_div=kl_div,
                gamma=self.gamma,
                rmse=reconstruction_metrics["rmse"],
            )
            logs.update(logs)
        return loss, logs


class SemiVFAE(ModelMixin, LightningModule):
    def __init__(self, dconfig: DatasetConfig, econfig: ExperimentConfig):
        """
        Args:
            context_dim: dimensionality or number of classes of the context variable
            beta: weight for classification loss
            gamma: weight for MMD
            num_features: number of random feature for fast MMD
            num_samples: number of samples for Monte Carlo estimation
            reconstruction_error: criteria for reconstruction error
            pred_transform: make function input into unormalized scores (logits)

        References:
            https://github.com/yevgeni-integrate-ai/VFAE/blob/master/VFAE_blog.ipynb
        """
        ModelMixin.__init__(self, dconfig, econfig)
        LightningModule.__init__(self)
        self.save_hyperparameters(
            # ignore=["prediction_loss", "reconstruction_loss", "mmd_loss", "klz2_loss", "mvn_diff_loss"]
        )

        mconfig = self.econfig.model
        self.context_dim = dconfig.context_dim
        self.target_dim = dconfig.target_dim
        self.target_type = dconfig.target_dim
        self.latent_dim = mconfig.latent_dim
        self.beta = float(mconfig.beta)
        self.gamma = float(mconfig.gamma)
        self.num_samples = mconfig.num_samples  # TODO
        self.pred_transform = self.init_pred_transform()
        self.activation_fn = make_activation_fn(mconfig.nonlinearity)
        self.lr = mconfig.lr

        # Modules
        self.qz1_xs = self.init_prob_cond_encoder(
            input_shape=dconfig.input_shape,
            context_dim=dconfig.context_dim,
            hidden_dims=mconfig.qz1_xs.hidden_dims,
            latent_dim=mconfig.latent_dim,
            nonlinearity=mconfig.nonlinearity,
        )
        self.qz2_z1y = self.init_prob_cond_encoder(
            input_shape=(mconfig.latent_dim,),
            context_dim=dconfig.target_dim,
            hidden_dims=mconfig.qz2_z1y.hidden_dims,
            latent_dim=mconfig.latent_dim,
            nonlinearity=mconfig.nonlinearity,
        )
        self.pz1_z2y = self.init_prob_cond_encoder(
            input_shape=(mconfig.latent_dim,),
            context_dim=dconfig.target_dim,
            hidden_dims=mconfig.pz1_z2y.hidden_dims,
            latent_dim=mconfig.latent_dim,
            nonlinearity=mconfig.nonlinearity,
        )
        self.px_z1s = self.init_cond_decoder(
            latent_dim=mconfig.latent_dim,
            context_dim=self.context_dim,
            hidden_dims=mconfig.px_z1s.hidden_dims,
            output_shape=self.dconfig.input_shape,
            nonlinearity=mconfig.nonlinearity,
        )
        self.qy_z1 = self.init_prob_predictor(
            input_shape=(mconfig.latent_dim,),
            hidden_dims=mconfig.qy_z1.hidden_dims,
            target_dim=dconfig.target_dim,
            nonlinearity=mconfig.nonlinearity,
        )
        self.py = self.init_target_prior(
            target_type=dconfig.target_type, target_probs=dconfig.target_probs
        )

        # Losses
        # KL(q(y|z1) || p(y)))
        self.reconstruction_loss = self.init_reconstruction_loss()
        self.prediction_loss = self.init_prediction_loss()
        self.klz2_loss = KLDivergence("mvn_diag", "mvn_std")  # KL(q(z2|z1, y) || p(z2))
        self.mvn_diff_loss = LogMVNDiffLoss(
            "mvn_diag", "mvn_diag"
        )  # log p(z1|z2,y) - log q(z1|x,s)
        self.mmd_loss = self.init_mmd_loss()

        # Metrics
        metrics = self.init_metrics()
        self.metrics = self.configure_metrics(metrics) if metrics else dict()

    def forward(self, x, s):
        """Generate representations"""
        return self._encode_unlabeled(x, s)[0]  # z1

    def training_step(self, batches, batch_idx):
        sbatch, ubatch = batches
        sbatch_size, ubatch_size = sbatch[0].size(0), ubatch[0].size(0)

        # Labeled step
        z1s, sloss, slogs = self._labeled_step(sbatch)

        # Unlabeled step
        z1u, uloss, ulogs = self._unlabeled_step(ubatch)

        # MMD
        mmd_loss = torch.tensor(0.0)
        if self.mmd_loss is not None:
            s = torch.concat([sbatch[2], ubatch[2]], dim=0)
            z1 = torch.concat([z1s, z1u], dim=0)
            mmd_loss = self.mmd_loss(z1[s == 0], z1[s == 1])

        # Logging
        with torch.no_grad():
            logs = OrderedDict(
                beta=self.beta,
                gamma=self.gamma,
                mmd_loss=mmd_loss,
            )

            if "train" in self.metrics:
                (
                    prediction_metrics,
                    fairness_metrics,
                    reconstruction_metrics,
                ) = self._compute_metrics("train")
                logs.update(
                    OrderedDict(
                        ac=prediction_metrics["accuracy"],
                        dp=fairness_metrics["discrimination"],
                        eo=fairness_metrics["equalized_odds"],
                        eg=fairness_metrics["error_gap"],
                        rmse=reconstruction_metrics["rmse"],
                    )
                )

            slogs.update(logs)
            self.log_dict(
                {f"L-train/{k}": v for k, v in slogs.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                sync_dist=False,
                batch_size=sbatch_size,
            )

            ulogs.update(logs)
            self.log_dict(
                {f"U-train/{k}": v for k, v in ulogs.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                sync_dist=False,
                batch_size=ubatch_size,
            )

        batch_size = sbatch_size + ubatch_size
        weight = max(ubatch_size / sbatch_size, 1)
        # print(weight, sloss.mean(), uloss.mean(), mmd_loss.mean())
        return (
            weight * sloss.mean() + uloss.mean()
        )  # +  batch_size * self.gamma * mmd_loss.mean()

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _, s = batch
        self.eval()
        (
            z1q,
            qz1mean,
            qz1logvar,
            z2,
            z2mean,
            z2logvar,
            pred_logits,
            qy_z1,
        ) = self._encode_unlabeled(x, s)
        z1p, pz1mean, pz1logvar, recon_logits = self._decode(z2, pred_logits, s)
        return {
            "z1": qz1mean,
            "z2": z2mean,
            "x": recon_logits,
            "y": pred_logits,
        }

    def configure_optimizers(self):
        params_non_frozen = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.Adam(params_non_frozen, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        lr_scheduler_config = dict(scheduler=scheduler, monitor="val/loss")
        return dict(optimizer=opt, lr_scheduler=lr_scheduler_config)

    @staticmethod
    def configure_metrics(metrics):
        new_metrics = dict(train={}, val={})
        for k, evaluator in metrics.items():
            new_metrics["train"][k] = evaluator.clone()
            new_metrics["val"][k] = evaluator.clone()
        return new_metrics

    def _update_metrics(self, x_logits, y_logits, x, y, s):
        stage = "train" if self.training else "val"
        if not self.metrics:
            return
        for k, e in self.metrics[stage].items():
            if k == "prediction":
                e.update(y_logits, y)
            elif k == "fairness":
                e.update(y_logits, y, s)
            elif k == "reconstruction":
                e.update(x_logits, x)

    def _compute_metrics(self, stage):
        prediction_metrics = self.metrics[stage]["prediction"].compute()
        fairness_metrics = self.metrics[stage]["fairness"].compute()
        reconstruction_metrics = self.metrics[stage]["reconstruction"].compute()
        return prediction_metrics, fairness_metrics, reconstruction_metrics

    def _eval_step(self, batch, prefix):
        _, loss, logs = self._labeled_step(batch)
        self.log_dict(
            {f"{prefix}/{metric_name}": value for metric_name, value in logs.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return logs

    def _encode_labeled(self, x, y, s):
        xs = combine_features(x, s)
        z1mean, z1logvar = self.qz1_xs(xs)
        z1 = self.qz1_xs.dsample(z1mean, z1logvar)

        z1y = combine_features(z1, y)
        z2mean, z2logvar = self.qz2_z1y(z1y)
        z2 = self.qz2_z1y.dsample(z2mean, z2logvar)

        if self.training:
            return z1, z1mean, z1logvar, z2, z2mean, z2logvar
        else:
            return z1mean, z1mean, z1logvar, z2mean, z2mean, z2logvar

    def _encode_unlabeled(self, x, s):
        xs = combine_features(x, s)
        z1mean, z1logvar = self.qz1_xs(xs)
        z1 = self.qz1_xs.dsample(z1mean, z1logvar)

        pred_logits, qy_z1 = self.qy_z1.point_and_dist(z1)
        z1y = combine_features(z1, pred_logits)
        z2mean, z2logvar = self.qz2_z1y(z1y)
        z2 = self.qz2_z1y.dsample(z1mean, z1logvar)

        if self.training:
            return z1, z1mean, z1logvar, z2, z2mean, z2logvar, pred_logits, qy_z1
        else:
            return (
                z1mean,
                z1mean,
                z1logvar,
                z2mean,
                z2mean,
                z2logvar,
                pred_logits,
                qy_z1,
            )

    def _decode(self, z2, y, s):
        z2y = combine_features(z2, y)
        z1mean, z1logvar = self.pz1_z2y(z2y)
        z1 = self.pz1_z2y.dsample(z1mean, z1logvar)

        z1s = combine_features(z2, s)
        recon_logits = self.px_z1s(z1s)

        if self.training:
            return z1, z1mean, z1logvar, recon_logits
        else:
            return z1mean, z1mean, z1logvar, recon_logits

    def _predict(self, z1):
        return self.qy_z1(z1).reshape(-1)

    def _labeled_step(self, batch):
        x, y, s = batch
        logs = dict()

        # Inference
        z1q, qz1mean, qz1logvar, z2, z2mean, z2logvar = self._encode_labeled(x, y, s)
        z1p, pz1mean, pz1logvar, recon_logits = self._decode(z2, y, s)
        pred_logits = self._predict(z1q)

        # Losses
        klz2_loss = self.klz2_loss(z2mean, z2logvar).mean()
        recon_loss = self.reconstruction_loss(recon_logits, x)
        mvn_diff_loss = self.mvn_diff_loss(
            z1q, qz1mean, qz1logvar, z1p, pz1mean, pz1logvar
        )
        pred_loss = self.prediction_loss(pred_logits, y)

        if isinstance(recon_loss, tuple):
            recon_loss, recon_logs = recon_loss
            logs.update(recon_logs)

        loss = klz2_loss + recon_loss + mvn_diff_loss + self.beta * pred_loss

        with torch.no_grad():
            self._update_metrics(recon_logits, pred_logits, x, y, s)
            logs.update(
                OrderedDict(
                    loss=loss,
                    recon_loss=recon_loss,
                    pred_loss=pred_loss,
                    klz2=klz2_loss,
                    mvn_diff_loss=mvn_diff_loss,
                )
            )
        return z1q, loss, logs

    def _unlabeled_step(self, batch):
        x, _y, s = batch
        logs = dict()

        # Inference
        (
            z1q,
            qz1mean,
            qz1logvar,
            z2,
            z2mean,
            z2logvar,
            pred_logits,
            qy_z1,
        ) = self._encode_unlabeled(x, s)
        pred = self.pred_transform(pred_logits)
        z1p, pz1mean, pz1logvar, recon_logits = self._decode(z2, pred, s)

        # Losses
        kly_loss = D.kl_divergence(qy_z1, self.py).mean()
        klz2_loss = self.klz2_loss(z2mean, z2logvar).mean()
        recon_loss = self.reconstruction_loss(recon_logits, x)
        mvn_diff_loss = self.mvn_diff_loss(
            z1q, qz1mean, qz1logvar, z1p, pz1mean, pz1logvar
        )

        if isinstance(recon_loss, tuple):
            recon_loss, recon_logs = recon_loss
            logs.update(recon_logs)

        loss = kly_loss + recon_loss + klz2_loss + mvn_diff_loss

        with torch.no_grad():
            self._update_metrics(recon_logits, pred_logits, x, _y, s)
            logs.update(
                OrderedDict(
                    loss=loss,
                    recon_loss=recon_loss,
                    kly=kly_loss,
                    klz2=klz2_loss,
                    mvn_diff=mvn_diff_loss,
                )
            )
        return z1q, loss, logs
