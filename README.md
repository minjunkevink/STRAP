# STRAP

This is the repository for the paper STRAP


## File Structure:
```bash
├── STRAP
│   ├── data/ # Folder to contain the data
│   │   ├── download_libero.py # Script to download libero datasets
│   ├── strap/
│   │   ├── retrieval/
│   │   │   ├── retrieval.py # Script to run retrieval on embeded datasets using the retrieval model
│   │   ├── embedding/
│   │   │   ├── encoders/ # Encoders for embedding
│   │   │   │   ├── encoders.py # Definitions of the different encoders
│   │   │   │   ├── encode_datasets.py # Script to encode a dataset using an encoder for retrieval.
│   │   │   ├── configs/
│   │   │   │   ├── libero_configs.py # Configs for the libero datasets
│   │   │   │   ├── libero_file_functions.py # File functions for the libero datasets
│   │   ├── README.md
│   ├── requirements.txt # Python requirements
```

## How to install
To install, run the following commands:
```bash
git clone https://github.com/WEIRDLabUW/STRAP.git
cd STRAP/strap
pip install -e .
```

## How to use
If you want to replicate our results on the LIBERO datasets, first download the datasets using the `download_libero.py` script. 
```bash
python data/download_libero.py
```
The next step is to encode the datasets using the `encode_datasets.py` script. Ours defaults to using both CLIP and DINOv2, and will embed both LIBERO_10 and LIBERO 90.
```bash
python strap/embedding/encode_datasets.py
```

Finally you are ready to run retrieval on the encoded datasets using the `retrieval.py` script. The default settings will select 3 demos from the LIBERO_10 "put both moka pots on the stove" task and retrieve the top 100 demos from the LIBERO_90 dataset. It has a minimum subtrajectory length of 20, and will use the DINOv2 agentview amera for retrieval.
```bash
python strap/retrieval/retrieval.py
```

You now have a retrieval dataset `put_both_moka_pots_retrieved_dataset.hdf5` in the `data/retrieval_results` folder! You can now use this dataset to train a policy using our 
[STRAP policy learning code](test). 

## Structure of the code
We designed STRAP to be modular and work on any HDF5 dataset that you can make a dataset config for. 

### Dataset Configs and HDF5 File Structure
The first step to defining a dataset is to define the HDF5 file structure. This is done by defining a `HDF5FileStructure` object. This object is a dataclass that contains the following fields:
- `demo_group`: The path from the root of the HDF5 file to the group containing all demonstrations. For example, if your demonstrations are stored under "/data/demos" in the HDF5 file, this would be "data/demos". Set to None if the demonstrations are stored directly at the root level.
- `obs_image_groups`: A list of paths from a demo group to image data. For example, if your images are stored at "/data/demos/###/images/rgb1" and "/data/demos/###/images/rgb2", this would be ["images/rgb1", "images/rgb2"]. These images will be used for embedding and retrieval.
- `obs_action_group`: The path from a demo group to action data. For example, if your actions are stored at "/data/demos/###/actions", this would be "actions".
- `obs_eef_pos_group`: The path from a demo group to end-effector position data. For example, if your end-effector positions are stored at "/data/demos/###/obs/ee_pos", this would be "obs/ee_pos".


The second step is to define a `DatasetConfig` object. This represents a dataset of multiple HDF5 files. This object is a dataclass that contains the following fields:
- `name`: The name of the dataset.
- `absolute_dataset_folder`: The path to the root folder containing all of the HDF5 files.
- `file_structure`: The `HDF5FileStructure` object.
- `ds_match_regex`: A regex pattern to match the HDF5 file paths in the dataset folder. Defaults to "*.hdf5".
- `embedding_extension`: What to append on to each file to save the embeddings version as. Defaults to "embeds.hdf5".
- `exclude_path`: An optional list of regexes to exclude from the dataset.
- `get_language_instruction`: Helper function to get the language instructions from the dataset.
- `save_trajectory_match`:  Helper function to save the retrieved trajectories to the output dataset.
- `initalize_save_file_metadata`: Helper function to copy over the metadata from the task dataset to the output dataset.

We have defined this for the LIBERO dataset in `strap/configs/libero_hdf5_config.py`.

#### Config Helper Functions

The `DatasetConfig` object also contains two helper functions you can define to use new datasets.

- `get_language_instruction`: A function that takes a h5py file in your dataset, and a demo_key and returns the language instruction for that demo. For example, in LIBERO this extracts the task name from the demo metadata.
- `save_trajectory_result`: A function that takes a h5py file input, a new h5py file as output, a `TrajectoryMatchResult`, the args passed for retrieval, the `DatasetConfig` of the input dataset, and the demo key to save the result under in the new file. This function must then handle copying over the relevant data from the TrajectoryMatchResult into the new retrieved dataset. We chose to define this as a configurable function in order to allow for different file structures for different datasets.
- `initalize_save_file_metadata`: A function that takes a h5py file input and the `DatasetConfig` of the input dataset. This function must then handle copying over the relevant metadata from the input dataset to the output dataset, and initializing the file.

## Configuring Embedding
We have defined the CLIP and DINOv2 encoders in `strap/embedding/encoders.py`. You can define your own encoders by inheriting from the `BaseEncoder` class and overwriting the `encode` and `preprocess` methods. 

To encode your datasets, use the `encode_datasets.py` script. This script will use the encoders you have defined in `get_encoders()` and the datasets you have defined in `get_datasets()`. Just change the dataset configs and the encoders list if you want to use different datasets or encoders. Set the `VERBOSE` flag to `False` to disable most of the logging statements. 

We also have a parallelized version of the dataset saving process as we noticed a large bottleneck in the dataset saving process. You can enable this by setting the `saver_threads` argument in the `encode_dataset` function to the number of threads you want to use. You can also set the `image_size` of the crop, and the batch_size of the encoder, as well as if you want to flip the images. LIBERO has upside down images, so we set `flip_images=True` for the embeddings to improve performance.



## Configuring Retrieval
The retrieval process is defined by the config created in the get_args() function. 
The parameters are:
- `task_dataset`: The `DatasetConfig` of the task dataset to use as the retrieval query.
- `offline_dataset`: The `DatasetConfig` of the dataset you want to retrieve from.
- `output_path`: The path to the output file.
- `model_key`: The key of the encoder to use for retrieval.
- `image_keys`: The keys of the images to use for retrieval. If it is multiple images, it will average the embeddings.
- `num_demos`: The number of demos to sample from the task dataset.
- `frame_stack`: How much to pad the sequence start by for models with frame stacks. In our libero experiments, we set this to 5, and then disable the robomimic padding.
- `action_chunk`: How much to pad the sequence end by for models with action chunking. In our libero experiments, we set this to 5.
- `top_k`: How many segments to retrieve.
- `task_dataset_filter`: A regex pattern or list of patterns to filter the demos in the task dataset.
- `offline_dataset_filter`: A regex pattern or list of patterns to filter the demos in the offline dataset.
- `min_subtraj_len`: The minimum length of a subtrajectory to create during slicing.
- `verbose`: Whether to print verbose logging statements.
- `retrieval_seed`: The seed to use for retrieval. Defaults to 42.

## Retrieval Results
The result hdf5 file will be saved to the path specified in the `output_path` argument. It will contain the information saved by your `save_trajectory_result` function. It also will contain the metadata at the root level of the file from the `task_dataset` 