# Federated Motor Imagery Classification for Privacy-Preserving Brain-Computer Interfaces

This is the PyTorch implementation of the following paper:

Tianwang Jia, Lubin Meng, Siyang Li, Jiajing Liu and Dongrui Wu. Federated Motor Imagery Classification for Privacy-Preserving Brain-Computer Interfaces.

## Usage

### Setup

Please see the `requirements.txt` for environment configuration.

```bash
pip install -r requirements.txt
```

### Datasets

Please use the `get_data.py` to download MI data from `moabb`.

```bash
cd ./data
python -u get_data.py
```

### Train and Test

Federated Learning approaches

Please use the following commands to train a model with federated learning strategy and perform leave-one-subject-out testing.

1. FedBS
python src/train.py --model eegnet --global_epochs 50 --local_epochs 2 --data_path "A:\Minor Project\FedBS\data\BNCI2014001" --fedbs True --rho 0.1

2. Fed_avg:
python src/train.py --model eegnet --global_epochs 50 --local_epochs 2 --data_path "A:\Minor Project\FedBS\data\BNCI2014001"

3. Fed_prox:
python src/train.py --model eegnet --global_epochs 50 --local_epochs 2 --data_path "A:\Minor Project\FedBS\data\BNCI2014001" --fedprox True


### Terms
FedAvg (Federated Averaging): The original, foundational FL algorithm. The server simply takes the mathematical average of the weights from all the client models.

FedProx: An upgraded algorithm designed for when clients have very different data (heterogeneous). It adds a penalty to prevent local models from drifting too far away from the global model.

SCAFFOLD: An algorithm that uses "control variates" to correct "client drift." It calculates the difference between the server's update direction and the client's update direction to keep everyone aligned.

MOON (Model-Contrastive FL): Uses contrastive learning to force the local model's internal logic to look similar to the global model, rather than memorizing its own local data.

FedFA (Federated Feature Augmentation): An algorithm that shares statistical features (not raw data) across clients to help models generalize better when data is skewed.

FedBS: The novel, custom algorithm invented by the authors of the paper you are replicating. They likely built a highly specialized aggregation method specifically optimized to handle the messy, noisy nature of Brain Signal (BS) datasets better than the standard algorithms.