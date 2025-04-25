import os
from strap.embedding.encoders import CLIP, DINOv2, PositionDifferenceEncoder
from strap.embedding.embedding_helper import embed_dataset
from strap.configs.libero_hdf5_config import get_libero_config
from strap.utils.constants import EMBEDDED_DATA_DIR

from tqdm.auto import tqdm

"""
Notes:
- If your embedding process crashes, there is a chance that the last embedding file written will be corrupted!
- If your process crashes, the logic to check if a file has already been processed might mistakenly skip some files!
"""

VERBOSE = True


def get_encoders(mode):
    """
    Returns encoders based on mode

    Args:
        mode (str): 'STRAP' or '2D'
    """
    if mode == "2D":
        # 2D tracking encoders
        models = [PositionDifferenceEncoder()]
    else:
        # Default STRAP encoders
        models = [
            DINOv2(model_class="facebook/dinov2-base", pooling="avg"),
            # CLIP(model_class="openai/clip-vit-base-patch16", pooling="avg", mm_vision_select_layer=-2),
        ]
    return models


def get_datasets(mode):
    """
    Returns dataset configs with appropriate embedding folders

    Args:
        mode (str): 'STRAP' or '2D'
    """
    # Get dataset configs with embedding folders
    libero_10 = get_libero_config("libero_10", mode)
    libero_90 = get_libero_config("libero_90", mode)
    
    # Make embedding directories
    os.makedirs(libero_10.embedding_folder, exist_ok=True)
    os.makedirs(libero_90.embedding_folder, exist_ok=True)
    
    return [libero_10, libero_90]


def embed_datasets(mode="STRAP"):
    """
    Embeds all datasets using the specified mode

    Args:
        mode (str): 'STRAP' or '2D'
    """
    datasets = get_datasets(mode)
    encoders = get_encoders(mode)

    print(f"\033[94mEmbedding using {mode} mode\033[0m")
    print(f"\033[94mEncoders: {[type(e).__name__ for e in encoders]}\033[0m")
    
    # LIBERO's images are upside down, so flip them
    flip_images = True
    print("\033[94m" + f"Flip imgs is {flip_images}" + "\033[0m")

    batch_size = 256
    image_size = (224, 224)

    for dataset in tqdm(datasets, desc="Embedding datasets", disable=VERBOSE):
        print(f"\033[94mEmbedding dataset: {dataset.name}\033[0m")
        print(f"\033[94m  Source: {dataset.absolute_dataset_folder}\033[0m")
        print(f"\033[94m  Target: {dataset.embedding_folder}\033[0m")
        
        embed_dataset(
            dataset,
            encoders,
            saver_threads=1,
            flip_images=flip_images,
            batch_size=batch_size,
            image_size=image_size,
            verbose=VERBOSE,
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed datasets using STRAP or 2D methods")
    parser.add_argument("--mode", type=str, default="STRAP", help="Either STRAP or 2D")
    args = parser.parse_args()
    
    embed_datasets(args.mode)
