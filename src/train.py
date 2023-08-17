import sys
import os

from typing import List
from pathlib import Path
from pprint import pprint, pformat
import os

from omegaconf import DictConfig, ListConfig, OmegaConf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import LightningLoggerBase

from .evaluators import save_results_json, save_representations
from .eval import eval
from . import utils


log = utils.get_logger(__name__)


# == Train & Evaluate ==================================================================================================
def train_and_evaluate(econfig: DictConfig):
    """
    Init objects from config, and train and evaluate model
    """

    # Setup ============================================================================================================
    print("Working directory : {}".format(os.getcwd()))
    # Set seed for random number generators in pytorch, numpy and python.random
    if econfig.get("seed"):
        pl.seed_everything(econfig.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{econfig.datamodule.target._target_}>")
    datamodule = hydra.utils.instantiate(econfig.datamodule.target)
    datamodule.setup()
    dm_kwargs = getattr(datamodule, "get_model_kwargs", {})()
    dconfig = utils.DatasetConfig(**dm_kwargs)
    log.info(f"Datamodule arguments:\n{pformat(dm_kwargs, indent=4)}")

    # Init lightning model
    log.info(f"Instantiating model <{econfig.model.target._target_}>")
    model = hydra.utils.instantiate(econfig.model.target, dconfig, econfig)
    log.info(f"Model architecture:\n{repr(model)}")

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in econfig:
        for name, cb_conf in econfig.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in econfig:
        for _, lg_conf in econfig.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{econfig.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        econfig.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=econfig,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model =================================================================================================
    log.info("Start training")
    dl = datamodule.train_dataloader()
    if not isinstance(dl, tuple):
        log.info(f"TRAIN={len(datamodule.train_dataloader().dataset)}")
    else:
        log.info(f"TRAIN={[len(d) for d in dl]} Batches")
    log.info(f"VAL={len(datamodule.val_dataloader().dataset)}")
    log.info(f"TEST={len(datamodule.test_dataloader().dataset)}")
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),  # Needed for model selection
    )
    log.info("End training")

    # Evaluation ======================================================================================================
    if econfig.get("eval_after_training"):
        log.info("Evaluation after training")
        # for ckpt in ["best", "last"]:
        for ckpt in ["best"]:
            checkpoint_path = Path(
                getattr(trainer.checkpoint_callback, f"{ckpt}_model_path")
            )

            if checkpoint_path:
                model = utils.load_model(
                    checkpoint_path, dconfig=dconfig, econfig=econfig, model=model
                )
                model.eval()
            else:
                raise FileNotFoundError("Model path is empty")

            metadata, results = eval(
                model=model,
                datamodule=datamodule,
                checkpoint_path=checkpoint_path,
                econfig=econfig,
            )

            log.info(f"Saving results for ckpt: {checkpoint_path}")
            save_results_json(f"{ckpt}.json", metadata, results, flatten=True)

            if econfig.get("save_representations"):
                log.info(f"Saving representations for ckpt: {checkpoint_path}")
                save_representations(
                    experiment_dir=Path("."),
                    checkpoint_path=checkpoint_path,
                    model=model,
                    dataloader=datamodule.test_dataloader(),
                )

        if "results" not in locals():
            log.info("No results")
        log.info(f"End evaluating ckpt: {checkpoint_path}")

    # Return metric score for hyperparameter optimization =============================================================
    optimized_metric = econfig.get("optimized_metric")
    if optimized_metric:
        if isinstance(optimized_metric, ListConfig) and len(optimized_metric) > 1:
            return [
                trainer.callback_metrics[metric].item() for metric in optimized_metric
            ]
        else:
            return trainer.callback_metrics[optimized_metric].item()


@hydra.main(config_path="../configs/", config_name="config", version_base=None)
def main(econfig: DictConfig):
    """Imports should be nested inside @hydra.main to optimize tab completion"""

    # import warnings; warnings.filterwarnings("error")
    os.environ["CUDA_DEVICE_ORDER"] = str(econfig.cuda_device_order)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(econfig.cuda_visible_devices)
    try:
        # Train models
        if econfig.dry_run:
            pprint(OmegaConf.to_yaml(econfig))
        else:
            return train_and_evaluate(econfig)
    except ValueError as ex:
        log.error(OmegaConf.to_yaml(econfig))
        utils.log_traceback(log, ex)
        log.error(">>>>>> ValueError <<<<<<<")
    except Exception as ex:
        log.error(OmegaConf.to_yaml(econfig))
        utils.log_traceback(log, ex)
        raise ex


if __name__ == "__main__":
    main()
