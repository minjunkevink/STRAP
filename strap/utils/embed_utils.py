from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
import threading
import h5py
import typing as tp


@dataclass
class DatasetInfo:
    key_to_len: tp.Dict[str, int]
    model_to_image_to_done: tp.Dict[str, tp.Dict[str, bool]]


@dataclass(eq=False)
class DatasetPointer:
    save_file: h5py.File
    save_path: str
    dataset_info: DatasetInfo
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        is_file = os.path.isfile(self.save_path)
        self.save_file = h5py.File(self.save_path, "a" if is_file else "w", swmr=True)
        if self.save_file.get("data") is None:
            self.save_file.create_group("data")

    def close(self):
        self.save_file.close()


def get_dataset_info(
    file: h5py.File,
    output_file: tp.Union[None, h5py.File],
    model_keys,
    image_keys,
    key_0="demo_1",
):
    """This method takes in the output file, creates the "data" group if it doesn't exist
    and returns the information used about the dataset for future use.

    Args:
        file (h5py.File): The input dataset file
        output_file (h5py.File): The output dataset file
        model_keys (_type_): all model keys of interest
        image_keys (_type_): all image keys of interest
        key_0 (str, optional): This is the key of the first demo to check if already embedded. Defaults to "demo_1".

    Returns:
        DatasetInfo: Information about the dataset
    """
    key_to_len = {k: len(file["data"][k]["actions"]) for k in file["data"].keys()}
    model_to_image_to_done = {}

    # If there is no output file:
    if output_file is None:
        model_to_image_to_done = {
            model_key: {image_key: False for image_key in image_keys}
            for model_key in model_keys
        }
        return DatasetInfo(key_to_len, model_to_image_to_done)

    if output_file.get("data") is None:
        grp = output_file.create_group("data")
    else:
        grp = output_file["data"]

    for model_key in model_keys:
        model_to_image_to_done[model_key] = {}
        for image_key in image_keys:
            if (
                (grp.get(key_0) is not None)
                and (grp[key_0].get(model_key) is not None)
                and (grp[key_0][model_key].get(image_key) is not None)
            ):
                model_to_image_to_done[model_key][image_key] = True
            else:
                model_to_image_to_done[model_key][image_key] = False
    return DatasetInfo(key_to_len, model_to_image_to_done)


def get_input_and_output_file_path(dataset_path):
    dataset_path_split = str(dataset_path).split("/")
    file_name = dataset_path_split[-1]

    dataset_path_out = "/".join(dataset_path_split[:-1])
    file_name_split = file_name.split(".")
    file_name_out = file_name_split[0] + "_embeds." + file_name_split[1]

    original_file_path = os.path.join(dataset_path_out, file_name)

    os.makedirs(dataset_path_out, exist_ok=True)
    dataset_out_file_path = os.path.join(dataset_path_out, file_name_out)
    return original_file_path, dataset_out_file_path


def get_all_datasets_info(datasets_paths, model_keys, img_keys):
    dataset_info = {}
    # Populate the dataset_info
    for dataset_path in datasets_paths:
        # prepare in and output paths
        original_file_path, dataset_out_file_path = get_input_and_output_file_path(
            dataset_path
        )

        is_file = os.path.isfile(dataset_out_file_path)

        with h5py.File(original_file_path, "r", swmr=True) as original_file:
            if not is_file:
                dataset_info[original_file_path] = get_dataset_info(
                    original_file, None, model_keys, img_keys
                )
            else:
                with h5py.File(dataset_out_file_path, "r", swmr=True) as save_file:
                    dataset_info[original_file_path] = get_dataset_info(
                        original_file, save_file, model_keys, img_keys
                    )

    return dataset_info
