# deepsc-st-pytorch
This repository contains code for JSAC 2021 "Semantic Communication Systems for Speech Transmission".
The code is based on pytorch implementation according to [the original code](https://github.com/Zhenzi-Weng/DeepSC-S) and the content of [the paper](https://ieeexplore.ieee.org/abstract/document/9450827).
## Project structureChannel layer
```bash
.
├── dataset                 # Dataset
├── channel.py              # Channel layer
├── config.py               # The config of channel
├── custom_dataset.py       
├── log                     # The folder where logs are saved
├── log.py                  # User-defined logger
├── make_dataset_t.py       # Make dataset
├── metrics.py              # Evaluation metrics of the model 
├── model_thesis.py         # model
├── README.md
├── results                 # The folder where model files and evaluation results are saved
│   └── saved_model_thesis_with_noise
│       ├── awgn
│       └── rayleigh
├── speech_processing.py    # Data preprocessing includes normalization, de-normalization, sampling, and de-sampling
├── test_thesis.py          # Model testing
├── train_thesis.py         # Model training
└── utils.py                # Other, include resampler, create folder, ...
```
## Prerequisites
* Python 3.8+ and [Conda](https://www.anaconda.com/)
* Environment
    ```bash
    conda create -n deepsc_s python=3.8
    conda activate deepsc_s
    python -m pip install -r requirements.txt
    ```
## Datasets
* The dataset adopted in this project can be found in [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791), Please download this dataset and put them into ./dataset/raw_data folder. Then do the following:
    ```bash
    mkdir dataset
    python make_dataset_t.py --sr 8000 --num_frame 128 --frame_size 0.016 --stride_size 0.016 --save_path ./dataset/deepsc/
    ```
* If you don't want to download the original dataset, you can use the data set we processed directly.
    ```
    mkdir dataset
    cd ./dataset
    
    git clone https://github.com/linhl1218/deepsc-dataset.git
    mv deepsc-dataset deepsc
    ```
## Usage
Example of testing model performance change under awgn channel SNR=10dB:
```bash
python test_thesis.py --channel_type awgn --snrs [10]
```
Train the model:
```bash
python train_thesis.py --channel_type awgn --num_epochs 400 --batch_size 16 --lr 0.001
```


