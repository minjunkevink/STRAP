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
    Represents the structure of an HDF5 file.
    """
    demo_group: tp.Union[str, None] = None # The keys to get to the group of demo data i.e "data/demos" that you can iterate over
    obs_image_groups: tp.List[str] = None # The keys to get to the group of observation images i.e "obs/left_camera_1" from inside a demo group
    obs_action_group: str = None # The key to get to the group of action data i.e "actions" from inside a demo group


@dataclass
class DatasetConfig:
    """
    Contains the entire configuration for a dataset of multiple files
    """
    name: str = None    # The name of the dataset
    absolute_dataset_folder: str  = None      # The absolute path to the folder containing the entire dataset
    file_structure: HDF5FileStructure = None    # Structure to access data
    file_paths: str = "*.hdf5"  # The regex path of the files to include in the dataset folder
    get_language_instruction: tp.Callable[[h5py.File, str], str] = None  # A function to get the language instruction from a passed hdf5 file and demo key
    # optional arguments
    embedding_extension: str = "embeds.hdf5" # The file ending of the embedded data
    exclude_path: tp.List[str] = field(default_factory=list) # Paths to exclude in dataset folder. Excluded if these strings in any part of the path
    
    
    
    def __post_init__(self):
        dataset_directory = Path(self.absolute_dataset_folder)
        dataset_paths = list(dataset_directory.rglob(self.file_paths))
        self.dataset_paths = []
        
        for path in dataset_paths:
            if len(self.exclude_path) < 1 or np.all([dex not in str(path) for dex in self.exclude_path]):
                # ignore the embedded file paths (since they match with a dataset)
                if self.embedding_extension in str(path):
                    continue
                self.dataset_paths.append(str(path))
        
        # replace the part after the last . with the embedding extension
        self.embedding_paths = [str(path).rsplit(".", 1)[0] + "_" + self.embedding_extension for path in self.dataset_paths]


#####################################################
# Code to help with embedding data  in parallel     #
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
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        is_file = os.path.isfile(self.save_path)
        
        if not is_file:
            # create an empty file. This helps with not having entirly corrupted files if the program crashes
            self.save_file = h5py.File(self.save_path, "w")
            self.save_file.close()
        
        self.save_file = h5py.File(self.save_path, "a")
        if self.save_file.get("data") is None:
            self.save_file.create_group("data")


    def close(self):
        self.save_file.close()



class DatasetSaver():
    def __init__(self, num_threads, verbose=True):
        self.verbose = verbose
        self.data_registry: tp.Dict[str, DatasetFilePointer]= {}
        self.data_registry_lock = threading.Lock()
        self.queue = []
        self.queue_lock = threading.Lock()
        
        # threading stuff:
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()

    
    
    def queue_save_job(self, save_path, dataset_info, embeddings, model_key, img_key, dataset_structure):
        # check if the dataset exists
        with self.data_registry_lock:
            if save_path not in self.data_registry:
                self.data_registry[save_path] = DatasetFilePointer(None, save_path, dataset_info)
        with self.queue_lock:
            self.queue.append((save_path, embeddings, model_key, img_key, dataset_structure))
        
    def save_job(self, job):
        save_path, embeddings, model_key, img_key, dataset_structure = job
        if self.verbose:
            print(f'saving job {model_key}, {img_key}, at path {save_path}')
        with self.data_registry_lock:
            data_pointer = self.data_registry[save_path]
        
        with data_pointer.lock:
            idx_ctr = 0
            grp = data_pointer.save_file[dataset_structure.demo_group] if dataset_structure.demo_group is not None else data_pointer.save_file
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
                    grp_model_key.create_dataset(img_key, data=embeddings[idx_ctr:idx_ctr + demo_len])
                except:
                    # del grp_new_k[img_key]
                    # grp_new_k.create_dataset(img_key, data=features[idx_ctr:idx_ctr + demo_len])
                    print(f"Error: {k} -> {model_key} -> {img_key}")
                # move idx
                idx_ctr += demo_len
        
        if self.verbose:
            print(f"saved {model_key}/{img_key}, at path {save_path}")
                
    
    def _worker(self):
        while not self.stop_event.is_set():
            job = None
            with self.queue_lock:
                if self.queue:
                    job = self.queue.pop(0)
            if job:
                self.executor.submit(self.save_job, job)
            else:
                self.stop_event.wait(timeout=0.1)  # Wait a bit before checking the queue again


    def wait_until_saved(self):
        while True:
            with self.queue_lock:
                length = len(self.queue)
            if length > 0:
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
