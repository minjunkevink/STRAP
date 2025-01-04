from strap.utils.file_utils import DatasetConfig, DatasetFileInfo, DatasetSaver, HDF5FileStructure
from strap.utils.processing_utils import HDF5Dataset
import typing as tp
from .encoders import BaseEncoder
import os
from tqdm.auto import tqdm
import h5py
from torch.utils.data import DataLoader

def embed_dataset(dataset: DatasetConfig, encoders: tp.List[BaseEncoder], saver_threads:int, flip_images=False, batch_size=32, image_size=(224,224), verbose: bool =False):
    """
    Embeds a dataset using the given encoders.
    
    Args:
        dataset (DatasetConfig): The dataset to embed
        encoders (List[BaseEncoder]): The encoders to use
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    # create a map of file path to work needed to be
    
    dataset_info: tp.Dict[str, DatasetFileInfo] = get_all_datasets_info(dataset.dataset_paths, encoders, verbose)

    # create the dataset saver
    saver = DatasetSaver(num_threads=saver_threads, verbose=verbose)
    
    for i in tqdm(range(len(dataset.dataset_paths)), desc=f"Embedding datasets"):
        # prepare in and ouput paths
        df_path, df_save_path = dataset.dataset_paths[i], dataset.embedding_paths[i]
        
        # process image keys
        for img_key in dataset.file_structure.obs_image_groups:
            if verbose:
                print(f"Embedding dataset: {df_path}, img_key: {img_key}")
            
            dataset = HDF5Dataset(df_path, img_key, img_size=image_size, flip_imgs=flip_images, verbose=verbose)

            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)

            # iter models
            for model in encoders:
                model_k = model.embedding_file_key
                
                # check if model embedding already exists i.e info -> file -> model_key -> img_key is True means it is done
                if dataset_info[df_path].model_to_image_to_done[model_k][img_key]:
                    if verbose:
                        print(model_k, img_key, "already exists. skipping...")
                    continue
                
                features = model.encode_dataloader(dataloader, verbose=1)

                if saver_threads > 1:
                    saver.queue_save_job(df_save_path, dataset_info[df_path], features, model_k, img_key)
                else:
                    # save one-by-one
                    if df_save_path not in saver.data_registry:
                        saver.data_registry[dataset_out_file_path] = DatasetPointer(None, dataset_out_file_path, dataset_info[original_file_path])
                    saver.save_job((dataset_out_file_path, features, model_k, img_key, model_kwargs, pair_skip))

    if saver_threads > 1:
        saver.wait_until_saved()
    
    
    

def get_all_datasets_info(dataset: DatasetConfig, encoders: tp.List[BaseEncoder], verbose):
    dataset_info = {}
    
    model_keys = [encoder.embedding_file_key for encoder in encoders]
    img_keys = dataset.file_structure.obs_image_groups
    if verbose:
        print(f"Using model keys to save: {model_keys}")
        print(f"Using image keys to embed: {img_keys}")
        

    # Populate the dataset_info
    for i in range(len(dataset.dataset_paths)):
        # prepare in and ouput paths
        df_path, df_save_path = dataset.dataset_paths[i], dataset.embedding_paths[i]
        is_file = os.path.isfile(df_path)
        
        with h5py.File(df_path, "r", swmr=True) as original_file:
            if not is_file:
                dataset_info[df_path] = get_dataset_file_info(original_file, None, model_keys, img_keys, dataset_structure=dataset.file_structure)
            else:
                with h5py.File(df_save_path, "r", swmr=True) as save_file:
                    dataset_info[df_path] = df_save_path(original_file, save_file, model_keys, img_keys, dataset_structure=dataset.file_structure)
    
    return dataset_info


def get_dataset_file_info(file: h5py.File, output_file: tp.Union[None, h5py.File], model_keys, image_keys, df_structure: HDF5FileStructure):
    """This method takes in the output file, creates the "data" group if it doesn't exist
    and returns the information used about the dataset for future use.

    Args:
        file (h5py.File): The input dataset file
        output_file (h5py.File): The output dataset file
        model_keys (_type_): all model keys of interist
        image_keys (_type_): all image keys of interest
        key_0 (str, optional): This is the key of the first demo to check if already embedded. Defaults to "demo_1".

    Returns:
        DatasetInfo: Information about the dataset
    """
    
    if df_structure.demo_group is None:
        key_0 = list(file.keys())[0]
        key_to_len = {k: len(file[k][df_structure.obs_action_group]) for k in file.keys()}
    else:
        key_0 = list(file[df_structure.demo_group].keys())[0]
        key_to_len = {k: len(file[df_structure.demo_group][k][df_structure.obs_action_group]) for k in file[df_structure.demo_group].keys()}


    model_to_image_to_done = {}
    
    # If there is no output file:
    if output_file is None:
        model_to_image_to_done = {model_key: {image_key: False for image_key in image_keys} for model_key in model_keys}
        return DatasetFileInfo(key_to_len, model_to_image_to_done)

    
    if output_file.get(df_structure.demo_group) is None and df_structure.demo_group is not None:
        grp = output_file.create_group(df_structure.demo_group)
    else:
        grp = output_file[df_structure.demo_group] if df_structure.demo_group is not None else output_file
        
    for model_key in model_keys:
        model_to_image_to_done[model_key] = {}
        for image_key in image_keys:
            model_to_image_to_done[model_key][image_key] = True
            for traj_key in grp.keys(): 
                if (grp.get(key_0) is None) or  (grp[key_0].get(model_key) is None) or (grp[key_0][model_key].get(image_key) is None):
                    model_to_image_to_done[model_key][image_key] = False
                    break
                
    return DatasetFileInfo(key_to_len, model_to_image_to_done)
