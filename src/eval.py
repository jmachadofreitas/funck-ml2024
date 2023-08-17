from argparse import ArgumentParser
from typing import Tuple, Union, List, Any
from pathlib import Path
import logging
import os

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import LightningLoggerBase

from .evaluators import create_metadata, save_results_json
from . import utils


log = utils.get_logger(__name__)


def eval(
    checkpoint_path: Path,
    econfig: DictConfig,
    model=None,
    datamodule=None,
):
    """Evaluate experiment"""
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    log.info("Start evaluation")

    ckpt_type = "last" if checkpoint_path.match("last.ckpt") else "best"

    if econfig.get("seed"):
        # Set seed for random number generators in pytorch, numpy and python.random
        pl.seed_everything(econfig.seed, workers=True)

    if datamodule is None:
        # Init lightning datamodule
        log.info(f"Instantiating datamodule <{econfig.datamodule.target._target_}>")
        datamodule = hydra.utils.instantiate(econfig.datamodule.target)
        datamodule.setup()

    if model is None:
        # Init lightning model
        log.info(f"Instantiating model <{econfig.model.target._target_}>")
        dm_kwargs = getattr(datamodule, "get_model_kwargs", {})()
        dconfig = utils.DatasetConfig(**dm_kwargs)
        model = utils.load_model(checkpoint_path, dconfig=dconfig, econfig=econfig)
        model.eval()

    metadata = create_metadata(econfig, model, datamodule, checkpoint_path, ckpt_type)

    # Evaluation dataloaders
    eval_dataloader = datamodule.test_dataloader()

    # Evaluators
    results: List[dict] = []
    # Evaluation Committee: Dataset specific evaluation
    if "evaluator" in econfig.datamodule:
        eval_conf = econfig.datamodule.evaluator
        log.info(f"Instantiating Evaluation Committee <{eval_conf._target_}>")
        out = hydra.utils.instantiate(eval_conf).evaluate(
            model, dataloader=eval_dataloader
        )
        results.append(out)

    return metadata, results


def main():
    import ray

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available experiments"
    )
    parser.add_argument("--experiments_dir", type=Path)
    parser.add_argument("-e", "--experiment", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument(
        "-p", "--num_cpus", default=20, type=int, help="Give preference to this version"
    )
    args = parser.parse_args()

    experiment_root = args.experiments_dir / args.experiment / args.dataset

    def find_best_ckpt_name(checkpoints_dir: Path):
        matches = sorted(checkpoints_dir.glob(f"epoch_*.ckpt"), key=os.path.getmtime)
        if matches:  # Pick the most recent version
            checkpoint_path = matches[-1]
            return checkpoint_path.name
        else:
            FileNotFoundError(checkpoints_dir)

    OmegaConf.register_new_resolver("hydra", lambda _: Path.cwd())

    def job(experiment_dir, config):
        if not OmegaConf.has_resolver("hydra"):
            OmegaConf.register_new_resolver("hydra", lambda _: Path.cwd())

        # for ckpt in ["best", "last"]:
        for ckpt in ["best"]:
            if ckpt == "best":
                checkpoint_name = find_best_ckpt_name(experiment_dir / "checkpoints")
            else:
                checkpoint_name = "last.ckpt"

            checkpoint_path = experiment_dir / "checkpoints" / checkpoint_name

            metadata, results = eval(checkpoint_path=checkpoint_path, econfig=config)

            log.info(f"Saving results for ckpt: {checkpoint_path}")
            save_results_json(
                experiment_dir / f"{ckpt}.json", metadata, results, flatten=True
            )

        if "results" not in locals():
            log.info("No results")

        return 0

    if not ray.is_initialized() and args.num_cpus:
        ray.init(num_cpus=args.num_cpus, local_mode=False)

    # Find configurations under the experiment root
    config_paths = list(experiment_root.glob("**/.hydra/config.yaml"))

    if not len(config_paths):
        raise FileNotFoundError("No configurations found")

    if args.list:
        for config_path in config_paths:
            print(config_path)
        return

    if args.num_cpus:
        job = ray.remote(job)

    # Run evaluation jobs
    jobs = list()
    for config_path in config_paths:
        log.info(f"Config path: '{config_path}'")
        experiment_dir = config_path.parent.parent
        config = OmegaConf.load(config_path)
        config.eval_after_training = True

        if args.num_cpus:
            jobs.append(
                job.remote(
                    experiment_dir=experiment_dir,
                    config=config,
                )
            )
        else:
            job(experiment_dir=experiment_dir, config=config)

    if args.num_cpus and jobs:
        ray.get(jobs)
        ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        utils.log_traceback(log, ex)
        raise ex
