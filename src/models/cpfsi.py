from collections import OrderedDict
from functools import reduce

import torch
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection
from omegaconf import OmegaConf, DictConfig

from ..utils import DatasetConfig, ExperimentConfig
from ..modules import (
    KLDivergence,
    FastMMD,
    combine_features,
    make_activation_fn,
    freeze,
)
from .mixin import ModelMixin


class CPFSI(ModelMixin, LightningModule):
    def __init__(self, dconfig: DatasetConfig, econfig: ExperimentConfig):
        """Conditional Privacy Funnel with Side-Information"""
        ModelMixin.__init__(self, dconfig, econfig)
        LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["reconstruction_loss", "prediction_loss"])

        mconfig = self.econfig.model
        self.context_dim = dconfig.context_dim
        self.latent_dim = mconfig.latent_dim
        self.alpha = float(mconfig.alpha)
        self.beta = float(mconfig.beta)
        self.gamma = float(mconfig.gamma)
        self.num_samples = mconfig.num_samples  # TODO
        self.lr = mconfig.lr
        self.activation_fn = make_activation_fn(mconfig.nonlinearity)

        # Modules
        self.encoder = self.init_prob_encoder(
            input_shape=dconfig.input_shape,
            hidden_dims=mconfig.encoder.hidden_dims,
            latent_dim=mconfig.latent_dim,
            nonlinearity=mconfig.nonlinearity,
        )
        self.decoder = self.init_cond_decoder(
            latent_dim=mconfig.latent_dim,
            context_dim=dconfig.context_dim,
            hidden_dims=mconfig.decoder.hidden_dims,
            output_shape=dconfig.input_shape,
            nonlinearity=mconfig.nonlinearity,
        )
        self.predictor = self.init_cond_predictor(
            input_shape=mconfig.latent_dim,
            context_dim=dconfig.context_dim,
            hidden_dims=mconfig.predictor.hidden_dims,
            target_dim=dconfig.target_dim,
            nonlinearity=mconfig.nonlinearity,
        )

        # Losses
        self.kl_loss = KLDivergence("mvn_diag", "mvn_std")
        self.reconstruction_loss = self.init_reconstruction_loss()
        self.prediction_loss = self.init_prediction_loss()
        self.mmd_loss = self.init_mmd_loss()

        # Metrics
        metrics = self.init_metrics()
        self.metrics = self.configure_metrics(metrics) if metrics else dict()

        # Optimizer
        if self.alpha == 0:
            freeze(self.decoder)
        if self.beta == 0:
            freeze(self.predictor)

    def forward(self, x, ctx=None):
        """Generate representations"""
        return self._encode(x)[0]

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
        z_mean, _, _ = self._encode(x)
        recon_logits = self._decode(z_mean, c)
        pred_logits = self._predict(z_mean, c)
        return {"z": z_mean, "x": recon_logits, "y": pred_logits}

    def configure_optimizers(self):
        params_non_frozen = filter(lambda p: p.requires_grad, self.parameters())
        opt = torch.optim.Adam(params_non_frozen, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        lr_scheduler_config = dict(scheduler=scheduler, monitor="val/loss")
        return dict(optimizer=opt, lr_scheduler=lr_scheduler_config)

    def _encode(self, x):
        loc, logscale = self.encoder(x)
        if self.training:  # Stochastic encoder
            sample = self.encoder.dsample(loc, logscale)
            return sample, loc, logscale
        else:  # Deterministic encoder
            return loc, loc, logscale

    def _decode(self, z, c):
        zc = combine_features(z, c)
        logits = self.decoder(zc)
        return logits

    def _predict(self, z, c):
        zc = combine_features(z, c)
        logits = self.predictor(zc)
        return logits.reshape(-1)

    def _train_step(self, batch, prefix="train"):
        loss, logs = self._step(batch)

        self.log_dict(
            {f"{prefix}/{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            sync_dist=False,
        )
        return loss

    def _eval_step(self, batch, prefix):
        """prefix: val/test"""
        loss, logs = self._step(batch)
        self.log_dict(
            {f"{prefix}/{metric_name}": value for metric_name, value in logs.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return logs

    def _step(self, batch):
        logs = dict()
        x, target, s = batch
        batch_size = x.size(0)

        # Inference
        z, mean, logvar = self._encode(x)
        recon_logits = self._decode(z, s)
        pred_logits = self._predict(z, s)

        # Losses
        recon_loss = self.reconstruction_loss(recon_logits, x)
        pred_loss = self.prediction_loss(pred_logits, target)
        kl_div = self.kl_loss(mean, logvar).mean()

        if isinstance(recon_loss, tuple):
            recon_loss, recon_logs = recon_loss
            logs.update(recon_logs)

        if isinstance(kl_div, tuple):
            kl_div, kl_logs = kl_div
            logs.update(kl_logs)

        mmd_loss = 0.0
        if self.mmd_loss is not None:
            mmd_loss += batch_size * self.gamma * self.mmd_loss(z[s == 0], z[s == 1])

        loss = kl_div + self.alpha * recon_loss + self.beta * pred_loss + mmd_loss

        with torch.no_grad():
            stage = "train" if self.training else "val"
            if stage in self.metrics:
                self._update_metrics(recon_logits, pred_logits, x, target, s)
                (
                    prediction_metrics,
                    fairness_metrics,
                    reconstruction_metrics,
                ) = self._compute_metrics(stage)
                logs.update(
                    OrderedDict(
                        ac=prediction_metrics["accuracy"],
                        dp=fairness_metrics["discrimination"],
                        eo=fairness_metrics["equalized_odds"],
                        eg=fairness_metrics["error_gap"],
                        rmse=reconstruction_metrics["rmse"],
                    )
                )

            logs.update(
                OrderedDict(
                    loss=loss,
                    recon_loss=recon_loss,
                    pred_loss=pred_loss,
                    kl_div=kl_div,
                    mmd_loss=mmd_loss,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
            )
        return loss, logs


class SemiCPFSI(CPFSI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batches, batch_idx):
        sbatch, ubatch = batches
        sbatch_size, ubatch_size = sbatch[0].size(0), ubatch[0].size(0)

        # Labeled step
        zs, sloss, slogs = self._labeled_step(sbatch)

        # Unlabeled step
        zu, uloss, ulogs = self._unlabeled_step(ubatch)

        # MMD
        mmd_loss = torch.tensor(0.0)
        if self.mmd_loss is not None:
            s = torch.concat([sbatch[2], ubatch[2]], dim=0)
            z = torch.concat([zs, zu], dim=0)
            mmd_loss = self.mmd_loss(z[s == 0], z[s == 1])

        # Logging
        with torch.no_grad():
            logs = OrderedDict(
                alpha=self.alpha,
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
        return (
            weight * sloss.mean()
            + uloss.mean()
            + batch_size * self.beta * mmd_loss.mean()
        )

    def _labeled_step(self, batch):
        x, target, s = batch
        logs = dict()

        # Inference
        z, mean, logvar = self._encode(x)
        recon_logits = self._decode(z, s)
        pred_logits = self._predict(z, s)

        # Losses
        recon_loss = self.reconstruction_loss(recon_logits, x)
        pred_loss = self.prediction_loss(pred_logits, target)
        kl_div = self.kl_loss(mean, logvar).mean()

        if isinstance(recon_loss, tuple):
            recon_loss, recon_logs = recon_loss
            logs.update(recon_logs)

        loss = kl_div + self.alpha * recon_loss + self.beta * pred_loss

        with torch.no_grad():
            self._update_metrics(recon_logits, pred_logits, x, target, s)
            logs.update(
                OrderedDict(
                    loss=loss,
                    recon_loss=recon_loss,
                    pred_loss=pred_loss,
                    kl_div=kl_div,
                )
            )
        return z, loss, logs

    def _unlabeled_step(self, batch):
        x, _y, s = batch
        logs = dict()

        # Inference
        z, mean, logvar = self._encode(x)
        recon_logits = self._decode(z, s)
        pred_logits = self._predict(z, s)

        # Losses
        recon_loss = self.reconstruction_loss(recon_logits, x)
        kl_div = self.kl_loss(mean, logvar).mean()

        if isinstance(recon_loss, tuple):
            recon_loss, recon_logs = recon_loss
            logs.update(recon_logs)

        loss = self.alpha * recon_loss

        with torch.no_grad():
            self._update_metrics(recon_logits, pred_logits, x, _y, s)
            logs.update(
                OrderedDict(
                    loss=loss,
                    recon_loss=recon_loss,
                    kl_div=kl_div,
                )
            )
        return z, loss, logs
