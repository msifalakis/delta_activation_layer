# Delta Activation Layer

Codebase for Delta Activation Layer  work (based on  Keras library with Tensorflow2).


## Setup

Default dataset path for the scripts is ```dataset_preparation/```  folder, This path is set in 

- ```video_generator/video_data_generator_UCF_resnet.py``` and 
  
- ```video_generator/video_data_generator_UCF_mobilenet.py```

and should be  be adjusted to your local dataset setup.

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

- ```video_generator/``` contains the video data generator and preprocessing scripts
- ```out/``` contains the trained models
- ```custom_layers.py``` Definitions of Delta_activation layers and L1 regularization layer

If your hardware setup has (like ours) multiple GPUs, you need to set the environment variable CUDA_VISIBLE_DEVICES to select  GPU for the experiments. This can be done either at the shell when calling the scripts, e.g 
```
	$ CUDA_VISIBLE_DEVICES=0 script.py
```
... **or** by setting  the environment accordingly from inside the script with the command ```os.environ["CUDA_VISIBLE_DEVICES"]="0"```. Our scripts contain this command, which will override any explicit setting at the shell. So either update the scripts with your selection or disable the command to allow choosing the GPU at the shell. 

In tensoflow2 the following LoCs in our scripts allow partial utilisation of GPU memory (which allows us to execute multiple scripts in one GPU). If this feature is not supported in your platform you need to disable (comment out #) these lines before executing the scripts.
```c++
  	physical_devices = tf.config.list_physical_devices('GPU')
 	tf.config.experimental.set_memory_growth(physical_devices[0], True)
```


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