from pathlib import Path
from collections import OrderedDict

import hydra
from omegaconf import DictConfig, OmegaConf

from .evaluators.reports import save_results_json
from . import utils


log = utils.get_logger(__name__)


def create_metadata(config, evaluator):
    metadata = OrderedDict(
        dataset=config.datamodule.name,
        datamodule=config.datamodule.target._target_,
        seed=config.seed,
        evaluator_cls=evaluator.__class__.__name__,
    )
    return metadata


# = Baseline ===========================================================================================================
def baseline(config: DictConfig):
    """Evaluate original datasets"""

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        utils.set_seed(config.seed)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.target._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule.target)
    datamodule.setup()

    # Evaluation dataloaders
    test_dataloader = datamodule.test_dataloader()

    # Evaluation Committee
    metadata, results = dict(), list()
    if "evaluator" in config.datamodule:
        eval_conf = config.datamodule.evaluator
        log.info(f"Instantiating Evaluation Committee <{eval_conf._target_}>")
        print(eval_conf)
        evaluator = hydra.utils.instantiate(eval_conf)
        if config.name == "baseline":
            # Baseline on the same evaluation dataset where representation are evaluated
            out = evaluator.baseline(test_dataloader=test_dataloader)
        else:
            raise NotImplementedError
        results.append(out)
        metadata = create_metadata(config, evaluator)

    log.info(f"Saving results for baseline")
    experiment_dir = Path(".")
    save_results_json(
        experiment_dir / f"baseline.json", metadata, results, flatten=True
    )

    return metadata, results


@hydra.main(config_path="../configs/", config_name="config", version_base="1.1")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    """ Imports should be nested inside @hydra.main to optimize tab completion """
    return baseline(config)


if __name__ == "__main__":
    main()
