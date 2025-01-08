from strap.embedding.encoders import CLIP, DINOv2
from strap.embedding.embedding_helper import embed_dataset
from tqdm.auto import tqdm

"""
Notes:
- If your embedding process crashes, there is a chance that the last embedding file being 
    written to will be corrupted.
- If your process crashes, the logic to check if a file has already been processed might mistakenly skip some files.
"""

VERBOSE = True

def get_encoders():
    """
    Overwrite this method in order to change which encoders are created/used in the embedding process.
    You can use this with your own custom encoders as well.
    """
    models = [
        CLIP(
            model_class="openai/clip-vit-base-patch16",
            pooling="avg",
            mm_vision_select_layer=-2,
        ),
        DINOv2(
            model_class="facebook/dinov2-base",
            pooling="avg"
        ),
    ]
    return models


def get_datasets():
    """
    Overwrite this method in order to change which datasets are encoded.
    You can use this with your own custom datasets as well.
    """
    from strap.configs.libero_hdf5_config import LIBERO_CONFIG, LIBERO_90_CONFIG, LIBERO_10_CONFIG
    return [LIBERO_90_CONFIG]

def embed_datasets():
    datasets = get_datasets()
    encoders = get_encoders()
    
    saver_threads = 1
    
    flip_images = False # Libero has upside down images
    batch_size = 1024
    image_size = (224,224)
    print("\033[94m" + f"Flip imgs is {flip_images}" + "\033[0m")

    
    for dataset in tqdm(datasets, desc="Embedding datasets", disable=VERBOSE):
        embed_dataset(  dataset, 
                        encoders,
                        saver_threads=saver_threads,
                        flip_images=flip_images, 
                        batch_size=batch_size,
                        image_size=image_size,
                        verbose=VERBOSE)

if __name__ == "__main__":
    embed_datasets()