# IMOCO4E-UNISS-SW107

SW-107 is a tool designed to train AI models for Anomaly Detection in the domain of Industrial Systems.
At present the training phase is tailored for the 
[One Year Industrial Component Degradation Dataset](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation?resource=download). 
Following the training phase, SW-107 allows for the use of the trained models to be used for inference in a system
using kafka brokers.

## Requirements
SW-107 requires [pyNeVer](https://github.com/NeVerTools/pyNeVer) and all its dependencies. We refer to its 
[Github Page](https://github.com/NeVerTools/pyNeVer) for more information regarding how to install them.

## How to use
The training phase can be executed launching [training_script.py](training_script.py) and requires two command line 
parameters:
- `--data_path`: Path to the folder containing the data. The raw data should be in the subfolder named raw/. The raw
data can be downloaded from Kaggle and the .csv must be copied in the raw/ subfolder.
- `--outputs_path`: Path to the folder which will contain the results of the training process. 
An example can be found in [outputs/](outputs/).

The inference capability can be executed launching [inference_script.py](inference_script.py) and requires five command line
parameters:
- `--bs`: Bootstrap Servers for Kafka Consumer and Producer.
- `--ct`: Topic of the Kafka Consumer.
- `--pt`: Topic of the Kafka Producer.
- `--config_path`: Path to config file. An example can be found in [configs/default_config.ini](configs/default_config.ini).
- `--log_perf`: File for logging script performances. If left to None the performances will not be logged.

## Important Notes

- More configuration parameters for the training process can be found and eventually re-defined in the file 
[utilities.py](utilities.py) and in particular in the function `training_config()`.
- [datasets.py](datasets.py) and [utilities.py](utilities.py) contain the python code needed to manage the data and
other utility functions and classes.