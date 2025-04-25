import typing as tp
import os
import argparse
import random
import numpy as np
from copy import deepcopy

from strap.utils.retrieval_utils import RetrievalArgs
from strap.utils.constants import RETRIEVAL_RESULTS_DIR
from strap.configs.libero_hdf5_config import get_libero_config
from strap.retrieval.retrieval_helper import run_retrieval, save_results


def get_args(parsed_args=None):
    """
    Returns retrieval arguments based on mode
    """
    # Determine mode
    mode = parsed_args.mode if parsed_args else "STRAP"
    
    # Set model info based on mode
    if mode == "2D":
        model_key = "PositionDiff"
        image_keys = "obs/ee_diff"
        model_type = "2D"
    else:  # Default to STRAP mode
        model_key = "DINOv2"
        image_keys = "obs/agentview_rgb"
        model_type = "STRAP"
    
    # Get dataset configs with appropriate embedding folders
    libero_10 = get_libero_config("libero_10", model_type)
    libero_90 = get_libero_config("libero_90", model_type)
    
    # Set output path
    output_dir = os.path.join(RETRIEVAL_RESULTS_DIR, mode)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_key}-stove-pot_retrieved_dataset.hdf5")

    return RetrievalArgs(
        task_dataset=libero_10,
        offline_dataset=libero_90,
        output_path=output_path,
        model_key=model_key,
        image_keys=image_keys,
        num_demos=5,
        frame_stack=5,
        action_chunk=5,
        top_k=100,
        task_dataset_filter=".*turn_on_the_stove_and_put_the_moka_pot_on_it.*",
        offline_dataset_filter=None,
        min_subtraj_len=20,
        retrieval_seed=42
    )


def main(args: RetrievalArgs):
    """Run retrieval"""
    print(f"Dataset paths:")
    print(f"  Base data: {args.task_dataset.absolute_dataset_folder}")
    if args.task_dataset.embedding_folder:
        print(f"  Embedding data: {args.task_dataset.embedding_folder}")
    print(f"  Embedding extension: {args.task_dataset.embedding_extension}")
    print(f"  Model key: {args.model_key}")
    print(f"  Image keys: {args.image_keys}")
    print(f"Output path: {args.output_path}")
    
    full_task_trajectory_results, retrieval_results = run_retrieval(args)
    save_results(args, full_task_trajectory_results, retrieval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose retrieval mode to be either STRAP or 2D")
    parser.add_argument("--mode", type=str, default="STRAP", help="Either STRAP or 2D")
    args = parser.parse_args()
    args = get_args(args)
    
    np.random.seed(args.retrieval_seed)
    random.seed(args.retrieval_seed)
    main(args)
