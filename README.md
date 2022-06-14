# Delta Activation Layer

Codebase for the Delta Activation Layer project (based on Keras library with Tensorflow2).

NOTE: This code is only experimental and therefore not comes with no liability (particularly in what relates to its quality or efficiency)


## Dataset setup

Default location of the dataset searched by the scripts in ```video_generator/``` is the ```dataset_preparation/```  folder.
This path is hardcoded in 

- ```video_generator/video_data_generator_UCF_resnet.py```, and 
  
- ```video_generator/video_data_generator_UCF_mobilenet.py```

It may therefore be altered to match your local dataset setup.

We use the whole dataset with *split 1*.

To prepare the dataset:

1. Download it from https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

2. Uncompress the rar file in the ```dataset_preparation/``` folder

3. Run ```dataset_preparation/split.py```to split the dataset into  train and test sets. This script also prepares frames out of each video files. (the script should be executed from inside ```dataset_preparation/``` where the dataset files are located). 

The format of the generated dataset folder shoulds look like this:

- ```dataset_preparation/UCF_train/<folder_per_class>```
- ```dataset_preparation/UCF_test/<folder_per_class>```
- ```dataset_preparation/UCF_train_frames/<folder_per_class>```
- ```dataset_preparation/UCF_test_frames/<folder_per_class>```



## General

- Dir ```video_generator/``` contains the video data generator and preprocessing scripts
- Dir ```out/``` contains the trained models
- File ```custom_layers.py``` contains the definitions of Delta_activation layers and L1 regularization layer
- Files ```*_model.py``` contain the various network models (that embed the DAL)
- Files ```*_fine_tune.py``` are top-level scripts to train a model starting from pre-trained weights
- Files ```*_inference.py``` are top-level scripts to run inference on trained models
- Files ```*_ORG_*.py``` involve baseline models in their original form from the literature (no DAL involved)



## Running with GPUs

If your hardware setup has (like ours) multiple GPUs, you may set the environment variable CUDA_VISIBLE_DEVICES to select  GPU for the experiments. This can be done either at the shell when calling the scripts, e.g 
```
	$ CUDA_VISIBLE_DEVICES=0 script.py
```
**or** by setting the environment programmatically inside the top-level scripts with the command ```os.environ["CUDA_VISIBLE_DEVICES"]="0"```.

Our scripts use the latter approach, which will thus override the value set in the former approach (setting the environment interactively at the shell prompt). So either update the scripts to reflect you GPU config, or disable the command in the scripts to make effective the GPU selection with the former approach. 

In tensoflow2 the following LoCs in our scripts make partial utilisation of GPU memory (which allows us to execute multiple scripts in one GPU). If this feature is not supported in your platform you need to disable (comment out #) these lines before executing the scripts.
```
  	physical_devices = tf.config.list_physical_devices('GPU')
 	tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## Library dependencies

This codebase has been tested with:

- python 3.8.x (3.8.6 to be more specific)
- tensorflow 2.3
- keras frontend 2.4.0 (the one included in Tensoflow 2.3)
- opencv-python 4.6.0.66
- psutil 5.9.1



## Running the experiments

These scripts should be run from inside the directory, where they are located. 

### ResNet-50

#### Baseline:

- Fine-tune (train) original network with UCF101 (orig. network was vanilla trained on ImageNet )
```
	$ UCF_ORG_ResNet50_fine_tune.py 
```

- Inference on UCF101 (uses ```UCF_TD_ResNet50_model.py``` and trained weights in the ```out/ORG_ResNet50_UCF_fine_tune/```)
```
	$ UCF_TD_ResNet50_inference.py
```


#### Spatial_sparsification:

- Train with L1_regularization layers and UCF101
```
	$ UCF_ResNet50_L1_fine_tune.py
```

- Inference with L1_regularization layers on UCF101 (uses ```UCF_ResNet50_L1_model.py``` with trained weights in the ```out/ResNet50_UCF_L1/```)
```
	$ UCF_ResNet50_L1_inference.py
```

#### Temporal_sparsification:

- Train with delta activation layers and UCF101
```
	$ UCF_ResNet50_delta_fine_tune.py
```

- Inference with delta activation layers on UCF101 (uses UCF_ResNet50_delta_model.py and trained weights in the ```out/ResNet50_UCF_delta/```)
```
	$ UCF_ResNet50_delta_inference.py
```


### MobileNet

#### Baseline

- Fine-tune (train) original network with UCF101
```
	$ UCF_ORG_MobileNet_fine_tune.py
```

- Inference on UCF101 (uses ```TD_mobilenet_model.py```  and trained weights in ```out/ORG_MobileNet_UCF_fine_tune/```)
```
	$ UCF_TD_MobileNet_inference.py
```

#### Temporal_sparsification

- Train with delta activation layers and UCF101
```
	$ mobilenet_delta_fine_tune.py
```

- Inference with delta activation layers on UCF101 (uses ```mobilenet_delta_model.py``` and trained weights in ```out/MobileNet_delta/```)
```
	$ mobilenet_delta_inference.py
```
