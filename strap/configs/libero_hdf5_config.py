import h5py
import json
from strap.utils.file_utils import HDF5FileStructure, DatasetConfig
from strap.utils.constants import REPO_ROOT


def get_libero_lang_instruction(f: h5py.File, demo_key) -> str:
    return json.loads(f["data"].attrs["problem_info"]).get("language_instruction", "dummy")

LIBERO_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{REPO_ROOT}/data/LIBERO",
    file_structure=HDF5FileStructure(
        demo_group="data",
        obs_image_groups=["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        obs_action_group="actions",
    ),
    file_paths="*.hdf5",
    embedding_extension="embeds.hdf5",
    get_language_instruction=get_libero_lang_instruction,
)

LIBERO_90_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{REPO_ROOT}/data/LIBERO",
    file_structure=HDF5FileStructure(
        demo_group="data",
        obs_image_groups=["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        obs_action_group="actions",
    ),
    file_paths="*.hdf5",
    embedding_extension="embeds.hdf5",
    exclude_path=["libero_10"],
    get_language_instruction=get_libero_lang_instruction,
)

LIBERO_10_CONFIG = DatasetConfig(
    name="libero",
    absolute_dataset_folder=f"{REPO_ROOT}/data/LIBERO",
    file_structure=HDF5FileStructure(
        demo_group="data",
        obs_image_groups=["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        obs_action_group="actions",
    ),
    file_paths="*.hdf5",
    embedding_extension="embeds.hdf5",
    exclude_path=["libero_90"],
    get_language_instruction=get_libero_lang_instruction,
)
