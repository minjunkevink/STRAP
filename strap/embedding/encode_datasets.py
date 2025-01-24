from strap.embedding.encoders import CLIP, DINOv2
from strap.embedding.embedding_helper import embed_dataset
from strap.configs.libero_hdf5_config import LIBERO_90_CONFIG, LIBERO_10_CONFIG

from tqdm.auto import tqdm

"""
Notes:
- If your embedding process crashes, there is a chance that the last embedding file written will be corrupted!
- If your process crashes, the logic to check if a file has already been processed might mistakenly skip some files!
"""

VERBOSE = True


def get_encoders():
    """
    Overwrite this method in order to change which encoders are created/used in the embedding process.
    You can use this with your own custom encoders as well.

    Returns:
        List[Encoder]: List of encoders to use for embedding
    """
    models = [
        DINOv2(model_class="facebook/dinov2-base", pooling="avg"),
        # CLIP(model_class="openai/clip-vit-base-patch16", pooling="avg", mm_vision_select_layer=-2),
        # ...
    ]
    return models


def get_datasets():
    """
    Overwrite this method in order to change which datasets are encoded.
    You can use this with your own custom datasets as well.

    Returns:
        List[DatasetConfig]: List of datasets to embed
    """

    return [LIBERO_90_CONFIG, LIBERO_10_CONFIG]


def embed_datasets():
    """
    Embeds all datasets in get_datasets() using the encoders in get_encoders()
    """

    datasets = get_datasets()
    encoders = get_encoders()

    flip_images = True  # Libero has upside down images
    print("\033[94m" + f"Flip imgs is {flip_images}" + "\033[0m")

    batch_size = 256
    image_size = (224, 224)

    for dataset in tqdm(datasets, desc="Embedding datasets", disable=VERBOSE):
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
    embed_datasets()
