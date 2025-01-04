from .encoders import CLIP, DINOv2
from .embedding_helper import embed_dataset
"""
Notes:
- If your embedding process crashes, there is a chance that the last embedding file being 
    written to will be corrupted.
- If your process crashes, the logic to check if a file has already been processed might mistakently skip some files.
"""

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
    from strap.configs.libero_hdf5_config import LIBERO_CONFIG
    return [LIBERO_CONFIG]

def embed_datasets():
    datasets = get_datasets()
    encoders = get_encoders()
    
    saver_threads = 1
    
    flip_imgs = False
    batch_size = 32
    image_size = (224,224)
    print("\033[94m" + f"Flip imgs is {flip_imgs}" + "\033[0m")

    
    for dataset in datasets:
        embed_dataset(  dataset, 
                        encoders,
                        saver_threads=saver_threads,
                        flip_imgs=flip_imgs, 
                        batch_size=batch_size,
                        image_size=image_size,
                        verbose=False)