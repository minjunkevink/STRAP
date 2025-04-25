import h5py
import json
import os

from strap.configs.libero_file_functions import initialize_libero_dataset
from strap.utils.file_utils import HDF5FileStructure, DatasetConfig
from strap.utils.constants import BASE_DATA_DIR, EMBEDDED_DATA_DIR
from strap.configs.libero_file_functions import get_libero_lang_instruction, save_trajectory_result_libero


LIBERO_FILE_STRUCTURE = HDF5FileStructure(
        demo_group="data",
        obs_image_groups=["obs/agentview_rgb"], # ["obs/agentview_rgb", "obs/eye_in_hand_rgb"]
        obs_action_group="actions",
        obs_eef_pos_group="obs/ee_pos",
)


def get_libero_config(subset, model_type=None):
    """
    Get DatasetConfig for LIBERO with correct paths
    
    Args:
        subset (str): 'libero_10' or 'libero_90'
        model_type (str, optional): Model type for embedding ('STRAP' or '2D')
            If provided, will set the embedding folder accordingly
    """
    
    # Set base data path
    base_dataset_folder = os.path.join(BASE_DATA_DIR, "base", "hdf5", subset)
    
    # Set embedded data path if model_type is provided
    embedding_folder = None
    if model_type:
        # Create embedding folder based on model type (STRAP or 2D)
        embedding_folder = os.path.join(EMBEDDED_DATA_DIR, f"{subset}_{model_type}")
        
    return DatasetConfig(
        name=subset,
        absolute_dataset_folder=base_dataset_folder,
        embedding_folder=embedding_folder,  # Will be None if model_type not provided
        file_structure=LIBERO_FILE_STRUCTURE,
        ds_match_regex="*.hdf5",
        embedding_extension="embeds.hdf5",
        get_language_instruction=get_libero_lang_instruction,
        save_trajectory_match=save_trajectory_result_libero,
        initalize_save_file_metadata=initialize_libero_dataset
    )


# Standard configs (original folder structure)
LIBERO_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{BASE_DATA_DIR}/LIBERO",
    file_structure=LIBERO_FILE_STRUCTURE,
    ds_match_regex="*.hdf5",
    embedding_extension="embeds.hdf5",
    get_language_instruction=get_libero_lang_instruction,
    save_trajectory_match=save_trajectory_result_libero,
    initalize_save_file_metadata=initialize_libero_dataset
)

LIBERO_90_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{BASE_DATA_DIR}/LIBERO",
    file_structure=LIBERO_FILE_STRUCTURE,
    ds_match_regex="*.hdf5",
    embedding_extension="embeds.hdf5",
    exclude_path=["libero_10"],
    get_language_instruction=get_libero_lang_instruction,
    save_trajectory_match=save_trajectory_result_libero,
    initalize_save_file_metadata=initialize_libero_dataset
)

LIBERO_10_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{BASE_DATA_DIR}/LIBERO",
    file_structure=LIBERO_FILE_STRUCTURE,
    ds_match_regex="*.hdf5",
    embedding_extension="embeds.hdf5",
    exclude_path=["libero_90"],
    get_language_instruction=get_libero_lang_instruction,
    save_trajectory_match=save_trajectory_result_libero,
    initalize_save_file_metadata=initialize_libero_dataset
)
