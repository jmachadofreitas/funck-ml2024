from typing import Mapping
from dataclasses import dataclass

# from omegaconf import OmegaConf, DictConfig
import torch

from ..utils import ExperimentConfig, DatasetConfig
from ..modules import *
from ..evaluators import *


class ModelMixin(object):
    def __init__(self, dconfig: DatasetConfig, econfig: ExperimentConfig):
        self._assert_params(econfig)
        self.dconfig = dconfig
        self.econfig = econfig
        self.mconfig = econfig.model

    def _assert_params(self, econfig):
        mconfig = econfig.model
        alpha = getattr(mconfig, "alpha", 0)
        beta = getattr(mconfig, "beta", 0)
        gamma = getattr(mconfig, "gamma", 0)
        assert alpha >= 0 and beta >= 0 and gamma >= 0

    def get_output_shape(self, module, *modules):
        x = torch.rand(1, *self.dconfig.input_shape)
        x = module(x)
        for mod in modules:
            x = mod(x)
        return x.shape[1:]

    def dtype2dist(self, dtype):
        if dtype == "bin":
            dist = "bernoulli"
        elif dtype == "cat":
            dist = "cat"
        elif dtype == "num":
            dist = "mvn_diag"
        else:
            raise NotImplementedError
        return dist

    def init_prob_layer(self, dist, block: nn.Module, input_dim: int, output_dim: int):
        """Init Probabilistic nn.Module Wrapper"""
        if dist == "mvn_diag":
            prob_layer = MultivariateNormalDiag(block, input_dim, output_dim)
        elif dist == "bernoulli":
            prob_layer = Bernoulli(block, input_dim, output_dim)
        else:
            raise NotImplementedError
        return prob_layer

    # Encoders ==========================================================================
    def init_pretrained_encoder(self, net: str):
        """
        Init encoder with pretrained features

        models:
            * AlexNet  https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        """
        latent_dim = self.mconfig.latent_dim
        pretrained = True
        encoder = None
        if net == "alexnet":
            alexnet_module = tv.models.alexnet(pretrained=pretrained)
            features = alexnet_module.features
            avgpool = alexnet_module.avgpool
            pre_latent = nn.Linear(256 * 6 * 6, latent_dim)

            # Freeze layers
            for param in features.parameters():
                param.requires_grad = False
            for param in avgpool.parameters():
                param.requires_grad = False

            encoder = nn.Sequential(features, avgpool, nn.Flatten(), pre_latent)
        elif net == "resnet18":
            resnet18 = tv.models.resnet18(pretrained=pretrained)

            # Freeze layers
            for p in resnet18.parameters():
                p.requires_grad = False

            # TODO
            # Unfreeze last layer
            for p in resnet18.layer4.parameters():
                p.requires_grad = True

            # Number of classes in our target task
            num_classes = 13

            # Number of conv features coming into the FC
            cnn_features = resnet18.fc.in_features
            print(f"cnn_features={cnn_features}")

            resnet18.fc = nn.Sequential(
                nn.Linear(cnn_features, 100, bias=True),
                nn.ReLU(),
                nn.Linear(100, num_classes, bias=True),
            )

            encoder = nn.Sequential(
                # ...
                nn.ReLU(),
                nn.Linear(100, num_classes, bias=True),
            )
        else:
            raise ValueError(f"Unknown model: '{net}'")
        return encoder

    # def _preamble(self, context_dim: int):
    #     norm_module = self.econfig.norm_module
    #     if self.econfig.hidden_dims is None:
    #         hidden_dims = list()
    #     else:
    #         hidden_dims = self.econfig.hidden_dims
    #     activation = "linear" if len(hidden_dims) == 0 else self.econfig.model.nonlinearity
    #
    #     if context_dim:  # Depends on combine_features
    #         input_shape = get_input_shape(self.dconfig.input_shape, context_dim)
    #     else:
    #         input_shape = self.dconfig.input_shape
    #     output_dim = self.dconfig.hidden_dims
    #     return input_shape, hidden_dims, output_dim, activation, norm_module

    def init_cond_encoder(
        self, input_shape, context_dim, hidden_dims, latent_dim, nonlinearity="relu"
    ):
        """
        Features are combined before passing to the conditional encoder

        Args:
            context_dim: used to create the init_encoder
        """
        if context_dim > 0:
            input_shape = get_input_shape(input_shape, context_dim)
        nonlinearity = "linear" if len(hidden_dims) == 0 else nonlinearity
        norm_module = False

        stride = 1
        kernel_size = 3
        if len(input_shape) > 1:
            if len(input_shape) == 3:  # conv2d, resnet2d, ...
                in_channels, in_width, in_height = input_shape
                conv_block = Conv2dBlock(
                    in_channels=in_channels,
                    out_channels=1,
                    kernel_size=kernel_size,
                    stride=stride,
                    hidden_channels=hidden_dims,
                    nonlinearity=nonlinearity,
                    norm_module=norm_module,
                )
                avgpool = nn.AdaptiveAvgPool2d((latent_dim, latent_dim))

            elif len(input_shape) == 2:  # conv1d, resnet1d, ...
                in_channels, in_dim = input_shape
                conv_block = Conv1dBlock(
                    in_channels=in_channels,
                    out_channels=1,
                    kernel_size=kernel_size,
                    stride=stride,
                    hidden_channels=hidden_dims,
                    nonlinearity=nonlinearity,
                    norm_module=norm_module,
                )
                avgpool = nn.AdaptiveAvgPool1d(2 * latent_dim)
            else:
                raise NotImplementedError(input_shape)

            conv_block = ResidualAdd(conv_block)
            hidden_output_dim = int(
                np.prod(get_output_shape(input_shape, conv_block, avgpool))
            )
            linear = nn.Linear(hidden_output_dim, latent_dim)

            # Init weights
            conv_block.apply(lambda p: init_weights(p, nonlinearity=nonlinearity))
            linear.apply(lambda p: init_weights(p, nonlinearity="linear"))

            encoder = nn.Sequential(conv_block, avgpool, nn.Flatten(), linear)
            return encoder
        else:
            input_dim = input_shape[0]
            encoder = MLPBlock(
                input_dim=input_dim, hidden_dims=hidden_dims, output_dim=latent_dim
            )
            encoder.apply(lambda p: init_weights(p, nonlinearity=nonlinearity))
            return encoder

    def init_encoder(self, input_shape, hidden_dims, latent_dim, nonlinearity="relu"):
        return self.init_cond_encoder(
            input_shape,
            context_dim=0,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            nonlinearity=nonlinearity,
        )

    def init_prob_cond_encoder(
        self, input_shape, context_dim, hidden_dims, latent_dim, nonlinearity="relu"
    ):
        cond_encoder = self.init_cond_encoder(
            input_shape, context_dim, hidden_dims, latent_dim, nonlinearity=nonlinearity
        )
        activation_fn = make_activation_fn(nonlinearity)
        block = nn.Sequential(cond_encoder, activation_fn)
        prob_cond_encoder = self.init_prob_layer(
            "mvn_diag", block=block, input_dim=latent_dim, output_dim=latent_dim
        )
        return prob_cond_encoder

    def init_prob_encoder(
        self, input_shape, hidden_dims, latent_dim, nonlinearity="relu"
    ):
        prob_encoder = self.init_prob_cond_encoder(
            input_shape,
            context_dim=0,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            nonlinearity=nonlinearity,
        )
        return prob_encoder

    # Decoders ==========================================================================
    def init_cond_decoder(
        self, latent_dim, context_dim, hidden_dims, output_shape, nonlinearity="relu"
    ):
        nonlinearity = "linear" if len(hidden_dims) == 0 else nonlinearity
        norm_module = False

        input_dim = latent_dim + context_dim
        if len(output_shape) > 1:
            stride = 1
            kernel_size = 3
            if len(output_shape) == 3:
                c, h, w = output_shape
                input_shape = (1, h, w)
                linear = nn.Linear(input_dim, np.prod(input_shape), bias=True)
                deconv_block = ConvTranspose2dBlock(
                    in_channels=1,
                    out_channels=c,
                    kernel_size=kernel_size,
                    stride=stride,
                    hidden_channels=hidden_dims,
                    nonlinearity=nonlinearity,
                    norm_module=norm_module,
                )

            elif len(output_shape) == 2:
                c, d = output_shape
                input_shape = (1, d)

                linear = nn.Linear(input_dim, np.prod(input_shape), bias=True)
                linear.apply(lambda p: init_weights(p, nonlinearity="linear"))

                deconv_block = ConvTranspose1dBlock(
                    in_channels=1,
                    out_channels=c,
                    kernel_size=kernel_size,
                    stride=stride,
                    hidden_channels=hidden_dims,
                    nonlinearity=nonlinearity,
                    norm_module=norm_module,
                )
            else:
                raise NotImplementedError(output_shape)

            deconv_block = ResidualAdd(deconv_block)
            linear.apply(lambda p: init_weights(p, nonlinearity="linear"))
            deconv_block.apply(lambda p: init_weights(p, nonlinearity=nonlinearity))
            decoder = nn.Sequential(linear, nn.Unflatten(1, input_shape), deconv_block)
        else:
            output_dim = output_shape[0]
            mlp_block = MLPBlock(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                nonlinearity=nonlinearity,
                norm_module=norm_module,
            )

            decoder = mlp_block
            decoder.apply(lambda p: init_weights(p, nonlinearity=nonlinearity))
        return decoder

    def init_decoder(self, latent_dim, hidden_dims, nonlinearity="relu"):
        return self.init_cond_decoder(
            latent_dim,
            context_dim=0,
            hidden_dims=hidden_dims,
            nonlinearity=nonlinearity,
        )

    def init_prob_cond_decoder(
        self, context_dim, output_shape, output_type, nonlinearity="relu"
    ):
        if len(output_shape) > 1:
            raise NotImplementedError
        cond_decoder = self.init_cond_decoder(context_dim=context_dim)
        activation_fn = self.make_activation_fn()
        dist = self.dtype2dist(output_type)
        block = nn.Sequential(cond_decoder, activation_fn)
        prob_cond_decoder = self.init_prob_layer(
            dist, block=block, input_dim=output_shape[0], output_dim=output_shape[0]
        )
        return prob_cond_decoder

    # Predictors ==========================================================================
    def init_cond_predictor(
        self,
        input_shape,
        context_dim,
        hidden_dims,
        target_dim,
        hidden_channels=None,
        nonlinearity="relu",
    ):
        nonlinearity = "linear" if len(hidden_dims) == 0 else nonlinearity
        norm_module = False

        if isinstance(input_shape, int) == 1:
            input_shape = (input_shape,)  # Make it a tuple

        if context_dim:
            input_shape = get_input_shape(input_shape, context_dim)

        if hidden_channels is None:
            hidden_channels = [input_shape[0]]
        out_channels = hidden_channels.pop(-1)

        if hidden_dims is None:
            hidden_dims = list()

        if len(input_shape) > 1:
            stride = 1
            kernel_size = 3
            if len(input_shape) == 3:
                in_channels, in_width, in_height = input_shape
                conv_block = Conv2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    hidden_channels=hidden_channels,
                    nonlinearity=nonlinearity,
                    norm_module=norm_module,
                )
            elif len(input_shape) == 2:
                in_channels, in_depth = input_shape
                conv_block = Conv1dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    hidden_channels=hidden_channels,
                    nonlinearity=nonlinearity,
                    norm_module=norm_module,
                )
            else:
                raise NotImplementedError
            conv_block = ResidualAdd(conv_block)
            conv_block.apply(lambda p: init_weights(p, nonlinearity=nonlinearity))

            mlp = MLPBlock(
                input_dim=np.prod(get_output_shape(input_shape, conv_block)),
                hidden_dims=hidden_dims,
                output_dim=target_dim,
                activation_fn=activation_fn,
                norm_module=norm_module,
            )

            mlp.apply(lambda p: init_weights(p, nonlinearity=nonlinearity))
            module = nn.Sequential(conv_block, nn.Flatten(), mlp)
        elif len(input_shape) == 1:
            (input_dim,) = input_shape
            mlp_block = MLPBlock(
                input_dim=input_dim, hidden_dims=hidden_dims, output_dim=target_dim
            )
            module = mlp_block
        else:
            raise NotImplementedError

        return module

    def init_predictor(
        self,
        input_shape,
        hidden_dims,
        target_dim,
        hidden_channels=None,
        nonlinearity="relu",
    ):
        return self.init_cond_predictor(
            input_shape,
            context_dim=0,
            hidden_dims=hidden_dims,
            target_dim=target_dim,
            hidden_channels=hidden_channels,
            nonlinearity=nonlinearity,
        )

    def init_prob_cond_predictor(
        self,
        input_shape,
        context_dim,
        hidden_dims,
        target_dim,
        hidden_channels=None,
        nonlinearity="relu",
    ):
        cond_predictor = self.init_cond_predictor(
            input_shape,
            context_dim,
            hidden_dims,
            target_dim,
            hidden_channels=hidden_channels,
            nonlinearity=nonlinearity,
        )
        target_dim = self.dconfig.target_dim
        target_type = self.dconfig.target_type
        nonlinearity = self.mconfig.nonlinearity
        dist = self.dtype2dist(target_type)
        activation_fn = make_activation_fn(nonlinearity)
        block = nn.Sequential(cond_predictor, activation_fn)
        prob_cond_predictor = self.init_prob_layer(
            dist, block=block, input_dim=target_dim, output_dim=target_dim
        )
        return prob_cond_predictor

    def init_prob_predictor(
        self,
        input_shape,
        hidden_dims,
        target_dim,
        hidden_channels=None,
        nonlinearity="relu",
    ):
        return self.init_prob_cond_predictor(
            input_shape,
            context_dim=0,
            hidden_dims=hidden_dims,
            target_dim=target_dim,
            hidden_channels=hidden_channels,
            nonlinearity=nonlinearity,
        )

    def init_pred_transform(self):
        target_type = self.dconfig.target_type
        if target_type == "bin":
            return torch.sigmoid
        elif target_type == "cat":
            return torch.softmax
        elif target_type == "num":
            return lambda t: t

    # Priors ==========================================================================================================
    def init_target_prior(self, target_type, target_probs):
        if target_type == "bin":
            if isinstance(target_probs, list):
                probs = target_probs[0]
            else:
                probs = target_probs
            dist = D.Bernoulli(probs=probs)
        elif target_type == "cat":  # Not used
            probs = target_probs
            dist = D.Categorical(probs=probs)
        elif target_type == "num":  # Not used
            loc = torch.zeros(dim, requires_grad=False)
            scale_diag = torch.ones(dim, requires_grad=False)
            dist = D.Independent(D.Normal(loc, scale_diag), 1)
        return dist

    # Losses ==========================================================================================================
    def init_reconstruction_loss(self, reduction="mean"):
        """All losses accept logits"""
        input_shape = self.dconfig.input_shape
        input_type = self.dconfig.input_type
        num_idxs = self.dconfig.num_idxs
        cat_idxs = self.dconfig.cat_idxs
        num_var = self.dconfig.num_var
        loss = None
        if len(input_shape) == 1:  # 1D
            if input_type == "mix":
                num_var = 1 if num_var is None else num_var
                loss = losses.TabularReconstructionLoss(
                    num_idxs, cat_idxs, num_var=num_var, reduction=reduction
                )
            if input_type == "bin":
                bce = nn.BCEWithLogitsLoss(reduction=reduction)
                loss = Lambda(lambda logits, target: bce(logits, target.float()))
            if input_type == "num":
                loss = nn.MSELoss(reduction=reduction)
            if input_type == "cat":
                loss = nn.CrossEntropyLoss(reduction=reduction)
        elif len(input_shape) > 1:
            loss = (
                nn.BCEWithLogitsLoss(reduction=reduction)
                if input_type == "bin"
                else nn.MSELoss(reduction=reduction)
            )
        else:
            raise ValueError(f"Invalid 'input_shape'={input_shape}")
        return loss

    def init_prediction_loss(self, reduction="mean"):
        target_type = self.dconfig.target_type
        if target_type == "bin":
            bce = nn.BCEWithLogitsLoss(reduction=reduction)
            return Lambda(lambda logits, target: bce(logits, target.float()))
        elif target_type == "cat":
            return nn.CrossEntropyLoss(reduction=reduction)
        elif target_type == "num":
            return nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError

    # def init_fairness_loss(self, reduction="mean"):
    #     target_type = self.dconfig.target_type
    #     context_type = self.dconfig.context_type
    #     if target_type == "bin" and context_type == "bin":
    #         return losses.DemographicParityLoss(reduction=reduction)
    #     elif target_type == "cat":
    #         loss = nn.CrossEntropyLoss(reduction=reduction)
    #         return MetricGapLoss(loss)
    #     elif target_type == "num":
    #         loss = nn.MSELoss(reduction=reduction)
    #         return MetricGapLoss(loss)
    #     else:
    #         raise NotImplementedError

    def init_mmd_loss(self):
        mconfig = self.econfig.model
        mmd_loss = None
        if self.gamma > 0:
            mmd_loss = FastMMD(
                num_features=mconfig.mmd_loss.num_features, gamma=mconfig.mmd_loss.gamma
            )
        return mmd_loss

    def init_metrics(self, recon_idx=None):
        # self.mconfig.metrics
        target_dim = self.dconfig.target_dim
        target_type = self.dconfig.target_type
        target_name = self.dconfig.target_name
        # Current support:
        context_dim = 1
        context_type = "bin"

        evaluators = dict()
        if recon_idx is None:
            recon_idx, recon_name, recon_dim, recon_type = 0, "x0", 1, "num"
            evaluators["reconstruction"] = ReconstructionEvaluator(
                recon_idx, recon_name, recon_dim, recon_type
            )
        evaluators["prediction"] = PredictionEvaluator(
            target_name, target_dim, target_type
        )
        if context_dim > 0:
            evaluators["fairness"] = FairnessEvaluator(
                target_name, target_dim, target_type, context_dim, context_type
            )
            # evaluators["privacy"] = PrivacyEvaluator(context_dim, context_type)   # Too slow: Needs SkEstimator
        return evaluators

    @staticmethod
    def configure_metrics(metrics):
        new_metrics = dict(train={}, val={})
        for k, evaluator in metrics.items():
            new_metrics["train"][k] = evaluator.clone()
            new_metrics["val"][k] = evaluator.clone()
        return new_metrics

    def _update_metrics(self, x_logits, y_logits, x, y, ctx):
        stage = "train" if self.training else "val"
        for k, e in self.metrics[stage].items():
            if k == "prediction":
                e.update(y_logits, y)
            elif k == "fairness":
                e.update(y_logits, y, ctx)
            elif k == "reconstruction":
                e.update(x_logits, x)

    def _compute_metrics(self, stage):
        prediction_metrics = self.metrics[stage]["prediction"].compute()
        fairness_metrics = self.metrics[stage]["fairness"].compute()
        reconstruction_metrics = self.metrics[stage]["reconstruction"].compute()
        return prediction_metrics, fairness_metrics, reconstruction_metrics
