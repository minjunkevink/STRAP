import typing as tp
from strap.utils.retrieval_utils import RetrievalArgs
from strap.utils.constants import REPO_ROOT
from copy import deepcopy
from retrieval_helper import run_retrieval, save_results
import numpy as np
import random

def get_args():
    from strap.configs.libero_hdf5_config import LIBERO_90_CONFIG, LIBERO_10_CONFIG
    return RetrievalArgs(
        task_dataset=deepcopy(LIBERO_10_CONFIG),
        offline_dataset=deepcopy(LIBERO_10_CONFIG),
        output_path=f"{REPO_ROOT}/data/retrieval_results/lib10.hdf5",
        model_key="facebook_dinov2-base_avg_None",
        image_keys="obs/agentview_rgb",
        num_demos=10,
        frame_stack=5,
        action_chunk=5,
        top_k=10,
        task_dataset_filter=".*put_both_moka.*",
        offline_dataset_filter=".*put_the.*",
        min_subtraj_len=20
    )



def main(args: RetrievalArgs):
    retrieval_results = run_retrieval(args)
    save_results(args, retrieval_results)

if __name__ == '__main__':
    args = get_args()
    # seed numpy and random
    np.random.seed(args.retrieval_seed)
    random.seed(args.retrieval_seed)
    main(args)