import re
import time
from concurrent.futures import ThreadPoolExecutor
import os
from dataclasses import dataclass, field
import threading
import typing as tp
from pathlib import Path
import h5py
import numpy as np


@dataclass
class HDF5FileStructure:
    """
    Holds structure of an HDF5 file.
    Args:
        demo_group (Union[str, None]): Key to the demo data, e.g., "{demo_group}/demo_0"
        obs_image_groups (List[str]): List of keys to the image observations from inside demo_group, e.g., obs_image_groups=["obs/left_camera_1", "obs/wrist_camera_1", ...] -> "{demo_group}/demo_0/{obs_image_groups[0]}"
        obs_action_group (str): Key to the action data, e.g., obs_action_group="actions" -> "{demo_group}/demo_0/{obs_action_group}"
        obs_eef_pos_group (str): Key to the endeffector position from inside demo_group, e.g., obs_eef_pos_group="obs/ee_pos" -> "{demo_group}/demo_0/{obs_eef_pos_group}", used for auto slicing
    """

    demo_group: tp.Union[str, None] = None
    obs_image_groups: tp.List[str] = None
    obs_action_group: str = None
    obs_eef_pos_group: str = None


def get_demo_grp(hdf5_file: h5py.File, demo_group_key: tp.Union[str, None]):
    if demo_group_key is None:
        return hdf5_file
    return hdf5_file[demo_group_key]


@dataclass
class DatasetConfig:
    """
    Holds the configuration for a dataset consisting of multiple files.
    Args:
        name (str): Name of the dataset
        absolute_dataset_folder (str): Absolute path to the folder containing the entire dataset
        file_structure (HDF5FileStructure): Structure to access data
        ds_match_regex (str): Regex to include files from the absolute_dataset_folder

        get_language_instruction (Callable[[h5py.File, str], str]): Function to get the language instruction from an hdf5 file and demo key
        save_trajectory_match (Callable): Function to save a single trajectory to the output file
        initalize_save_file_metadata (Callable): Function to initialize the output file's metadata

        embedding_extension (str, optional): The file suffix of the embedded data
        exclude_path (List[str], optional): Paths to exclude in dataset folder. Excluded strings in any part of the path
        embedding_folder (str, optional): Path to folder containing embedding files. If None, will use absolute_dataset_folder
    """

    name: str = None
    absolute_dataset_folder: str = None
    file_structure: HDF5FileStructure = None
    ds_match_regex: str = "*.hdf5"
    get_language_instruction: tp.Callable[[h5py.File, str], str] = None
    save_trajectory_match: tp.Callable = None
    initalize_save_file_metadata: tp.Callable = None
    embedding_extension: str = "{model_key}.hdf5"
    exclude_path: tp.List[str] = field(default_factory=list)
    embedding_folder: str = None

    def __post_init__(self):
        dataset_directory = Path(self.absolute_dataset_folder)
        dataset_paths = list(dataset_directory.rglob(self.ds_match_regex))
        self.dataset_paths = []

        for path in dataset_paths:
            if len(self.exclude_path) < 1 or np.all(
                [dex not in str(path) for dex in self.exclude_path]
            ):
                # ignore the embedded file paths (since they match with a dataset)
                if self.embedding_extension in str(path):
                    continue
                self.dataset_paths.append(str(path))

        # If embedding folder is specified, use that to construct embedding paths
        if self.embedding_folder:
            # Get just the filename (basename) of each dataset path
            base_dataset_names = [os.path.basename(path) for path in self.dataset_paths]
            # Construct embedding path with embedding folder and modified filename
            self.embedding_paths = [
                os.path.join(self.embedding_folder, path.rsplit(".", 1)[0] + "_" + self.embedding_extension)
                for path in base_dataset_names
            ]
        else:
            # Original behavior: create embedding paths in same folder as dataset
            self.embedding_paths = [
                str(path).rsplit(".", 1)[0] + "_" + self.embedding_extension
                for path in self.dataset_paths
            ]

    def filter_(self, regex_to_match: tp.List[str]):
        """
        This method removes all the paths that do not contain any of the regexes in regex_to_match in place.
        Args:
            regex_to_match (List[str]): List of regexes to match. Only one of the regexes needs to match for it to be included
        """
        # iterate backwards
        for i in range(len(self.dataset_paths) - 1, -1, -1):
            if not any(
                re.match(regex, self.dataset_paths[i]) for regex in regex_to_match
            ):
                del self.dataset_paths[i]
                del self.embedding_paths[i]

    def __len__(self):
        """
        Return:
            # of dataset files in this dataset
        """
        return len(self.dataset_paths)


#####################################################
# Code to help with embedding data in parallel     #
#####################################################


@dataclass
class DatasetFileInfo:
    key_to_len: tp.Dict[str, int]
    model_to_image_to_done: tp.Dict[str, tp.Dict[str, bool]]


@dataclass(eq=False)
class DatasetFilePointer:
    save_file: h5py.File
    save_path: str
    dataset_info: DatasetFileInfo
    demo_group_key: tp.Union[str, None] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        is_file = os.path.isfile(self.save_path)

        if not is_file:
            # create an empty file. This helps with not having entirely corrupted files if the program crashes
            self.save_file = h5py.File(self.save_path, "w")
            self.save_file.close()

        self.save_file = h5py.File(self.save_path, "a")
        if (
            self.demo_group_key is not None
            and self.save_file.get(self.demo_group_key) is None
        ):
            self.save_file.create_group(self.demo_group_key)

    def close(self):
        self.save_file.close()


class DatasetSaver:
    def __init__(self, num_threads, verbose=True):
        self.verbose = verbose
        self.data_registry: tp.Dict[str, DatasetFilePointer] = {}
        self.data_registry_lock = threading.Lock()
        self.queue = []
        self.queue_lock = threading.Lock()

        # threading
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()

    def queue_save_job(
        self, save_path, dataset_info, embeddings, model_key, img_key, dataset_structure
    ):
        # check if the dataset exists
        with self.data_registry_lock:
            if save_path not in self.data_registry:
                self.data_registry[save_path] = DatasetFilePointer(
                    None, save_path, dataset_info, dataset_structure.demo_group
                )
        with self.queue_lock:
            self.queue.append(
                (save_path, embeddings, model_key, img_key, dataset_structure)
            )

    def save_job(self, job):
        save_path, embeddings, model_key, img_key, dataset_structure = job
        with self.data_registry_lock:
            data_pointer = self.data_registry[save_path]

        with data_pointer.lock:
            idx_ctr = 0
            grp = (
                data_pointer.save_file[dataset_structure.demo_group]
                if dataset_structure.demo_group is not None
                else data_pointer.save_file
            )
            for k in data_pointer.dataset_info.key_to_len.keys():
                # check if 'demo' group exists or create new
                if grp.get(k) is None:
                    grp_k = grp.create_group(k)
                else:
                    grp_k = grp[k]

                # check if 'model' group exists or create new
                if grp_k.get(model_key) is None:
                    grp_model_key = grp_k.create_group(model_key)
                else:
                    grp_model_key = grp_k[model_key]

                demo_len = data_pointer.dataset_info.key_to_len[k]
                # store embedding of 'img_key' as dataset
                try:
                    grp_model_key.create_dataset(
                        img_key, data=embeddings[idx_ctr : idx_ctr + demo_len]
                    )
                except:
                    print(f"Error: {k} -> {model_key} -> {img_key}")
                # move idx
                idx_ctr += demo_len
        # flush file
        data_pointer.save_file.flush()
        if self.verbose:
            print(f"Saved {model_key}/{img_key}, at path {save_path}")

    def _worker(self):
        while not self.stop_event.is_set():
            job = None
            with self.queue_lock:
                if self.queue:
                    job = self.queue.pop(0)
            if job:
                self.executor.submit(self.save_job, job)
            else:
                # wait a bit before checking the queue again
                self.stop_event.wait(timeout=0.1)

    def wait_until_saved(self):
        while True:
            with self.queue_lock:
                length = len(self.queue)
            if length > 0:
                time.sleep(0.1)
                continue
            break
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.worker_thread.join()
        self.executor.shutdown(wait=True)
        for data_pointer in self.data_registry.values():
            data_pointer.close()

    def __del__(self):
        self.stop()
