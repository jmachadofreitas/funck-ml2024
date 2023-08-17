import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil

from tinydb import TinyDB
from uuid import uuid4
from hashlib import md5

import numpy as np
import pandas as pd

from . import utils


logger = utils.get_logger(__name__)


def get_experiment_uuid():
    """Universal Unique ID for each run"""
    return str(uuid4().hex)


def get_experiment_id(metadata):
    """Universal Unique ID for each run"""
    experiment_cte = [
        "dataset",
        "num_samples_per_class",
        "model",
        "alpha",
        "beta",
        "gamma",
        "latent_dim",
        "estimator",
    ]
    plaintext = "".join([str(v) for k, v in metadata.items() if k in experiment_cte])
    experriment_id = md5(plaintext.encode()).hexdigest()
    return experriment_id


def get_replicate_id(experiment_id, metadata):
    """
    Replicate ID -- Same ID for replicates of the same experiment.

    Definition:
        - replicate_cte: constant attributes in a replicate -> replicate definition
        - (Model + HParams + Dataset) - Seed

    Allows to aggregate over different seeds
    """
    replicate_cte = ["seed"]
    plaintext = experiment_id + "".join(
        [str(v) for k, v in metadata.items() if k in replicate_cte]
    )
    replicate_id = md5(plaintext.encode()).hexdigest()
    return replicate_id


def get_measure_id(replicate_id, result: dict):
    """
    Definition:
        evaluation: representation evaluation -> repeated measurements
    """
    evaluation_cte = ["fold"]
    plaintext = replicate_id + "".join([str(result[k]) for k in evaluation_cte])
    return md5(plaintext.encode()).hexdigest()


def extract(args):
    """Extract results (with defined extension) from multiple experiments to single directory"""
    if args.list:
        for p in args.experiment_root.iterdir():
            if p.is_dir():
                print(p)
        return

    # Run aggregation.
    for src in args.experiment_root.glob(f"**/*{args.suffix}"):
        dst = Path(args.output)
        for p in src.parts[-6:-2]:
            dst /= p
        dst /= src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(src, "->", dst)
        shutil.copy(src, dst)


def summarize(args):
    """
    Concatenate results from multiple experiments
    """

    logger.info(f"Experiment root: {args.experiment_root}")

    if args.list:
        for p in args.experiment_root.iterdir():
            if p.is_dir():
                print(p)
        return

    # Run aggregation.
    if args.filestem:
        filestem = args.filestem.strip()
        dst = args.experiment_root / f"{filestem}.tinydb"
    else:
        ckpt = args.ckpt.strip()
        dst = args.experiment_root / f"{ckpt}.tinydb"

    # Clean-up existing file before creating the new one
    logger.info(f"Removing {dst}")
    dst.unlink(missing_ok=True)

    # Create destination dir
    dst.parent.mkdir(parents=True, exist_ok=True)
    db = TinyDB(dst, indent=2)
    metadata_tbl = db.table("metadata")
    results_tbl = db.table("results")

    if args.filestem:
        filestem = args.filestem.strip()
        iterator = args.experiment_root.glob(f"**/*/{filestem}.json")
    else:
        ckpt = args.ckpt.strip()
        iterator = args.experiment_root.glob(f"**/*/{ckpt}.json")

    for src in iterator:
        print(src)
        with open(src, "r") as f:
            record = json.load(f)

        metadata = record["metadata"]
        results = record["results"]

        filepath: Path = src.parent / "predictor_weights.csv"
        if filepath.exists():
            with filepath.open("r") as f:
                predictor_weights = list(np.loadtxt(f).squeeze())
            metadata["predictor_weights"] = predictor_weights

        if len(results) == 0:
            raise ValueError("No results")

        metadata["results_path"] = str(src)

        # Universal Unique ID for each experiment
        experiment_uuid = get_experiment_uuid()
        experiment_id = get_experiment_id(metadata)
        metadata.setdefault("experiment_uuid", experiment_uuid)
        metadata.setdefault("experiment_id", experiment_id)

        # Same ID for experiment replicates (diff. seeds)
        replicate_id = get_replicate_id(experiment_id, metadata)
        [r.setdefault("experiment_uuid", experiment_uuid) for r in results]
        [r.setdefault("replicate_id", replicate_id) for r in results]

        for result in results:
            # Unique ID to aggregate different measurement (evaluation folds)
            measure_uuid = get_experiment_uuid()
            result.setdefault("measure_uuid", measure_uuid)
            # Unique ID to aggregate different measurement (evaluation folds)
            measure_id = get_measure_id(replicate_id, result)
            result.setdefault("measure_id", measure_id)

            # evaluation_id = get_evaluation_id(result)
            # result.setdefault("evaluation_id", evaluation_id)

        metadata_tbl.insert(metadata)
        results_tbl.insert_multiple(results)
    db.close()


def main():
    """See https://docs.python.org/3/library/argparse.html"""
    # Parent (must be fully initialized before passing it)
    parent_parser = ArgumentParser(description="The parent parser", add_help=False)
    parent_parser.add_argument(
        "-l", "--list", action="store_true", help="List available experiments"
    )
    parent_parser.add_argument(
        "-e", "--experiment_root", default="experiments/", type=Path
    )
    parent_parser.add_argument(
        "-c", "--ckpt", default="best", type=str, help="best.ckpt and last.ckpt"
    )

    # Main
    main_parser = ArgumentParser()
    subparsers = main_parser.add_subparsers(
        dest="command", required=False, metavar="summarize, extract"
    )
    summarize_parser = subparsers.add_parser("summarize", parents=[parent_parser])
    extract_parser = subparsers.add_parser("extract", parents=[parent_parser])

    # Summarize
    summarize_parser.add_argument(
        "-s",
        "--suffix",
        default=".tinydb",
        type=str,
        help="Output extension: .json, .csv, .tinydb, ...",
    )
    summarize_parser.add_argument(
        "-f", "--filestem", default="", type=str, help="E.g. baseline"
    )
    summarize_parser.set_defaults(func=summarize)

    # Extract
    extract_parser.add_argument(
        "-s", "--suffix", default=".png", type=Path, help="Extension: png, pdf, ..."
    )
    extract_parser.add_argument(
        "-o", "--output", default="outputs/", type=Path, help="Output dir"
    )
    extract_parser.set_defaults(func=extract)

    args = main_parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    main()
