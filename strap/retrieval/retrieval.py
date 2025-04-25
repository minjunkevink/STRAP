import typing as tp
from strap.utils.retrieval_utils import RetrievalArgs
from strap.utils.constants import REPO_ROOT
from copy import deepcopy
from strap.retrieval.retrieval_helper import run_retrieval, save_results
import numpy as np
import random
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run retrieval process with specified arguments.")
    parser.add_argument("--output_path", type=str, default="/media/kimkj/data/retrieved/stove-pot_retrieved_dataset.hdf5", help="Path to save the retrieved dataset.")
    parser.add_argument("--mode", type=str, default="STRAP", help="Either STRAP or 2D")




def get_args():

    from strap.configs.libero_hdf5_config import LIBERO_90_CONFIG, LIBERO_10_CONFIG

    if args.mode == "STRAP":
        output_path = f"/media/kimkj/data/retrieved/STRAP/IMG-stove-pot_retrieved_dataset.hdf5"
        model_key = "DINOv2"
        image_keys = "obs/agentview_rgb"
    elif args.mode == "2D":
        output_path = f"/media/kimkj/data/retrieved/2D/2D-stove-pot_retrieved_dataset.hdf5"
        model_key = "PositionDifference"
        image_keys = "obs/ee_diff"
    else:
        raise ValueError("Invalid mode specified. Choose either 'STRAP' or '2D'.")

    return RetrievalArgs(
        task_dataset=deepcopy(LIBERO_10_CONFIG),
        offline_dataset=deepcopy(LIBERO_90_CONFIG),
        output_path=output_path,
        model_key=model_key,
        image_keys=image_keys,
        num_demos=5,
        frame_stack=5,
        action_chunk=5,
        top_k=100,
        task_dataset_filter=".*turn_on_the_stove_and_put_the_moka_pot_on_it.*",
        offline_dataset_filter=None,
        min_subtraj_len=20
    )


def main(args: RetrievalArgs):
    full_task_trajectory_results, retrieval_results = run_retrieval(args)
    save_results(args, full_task_trajectory_results, retrieval_results)


if __name__ == "__main__":
    args = get_args()

    np.random.seed(args.retrieval_seed)
    random.seed(args.retrieval_seed)
    main(args)
