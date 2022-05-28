# Source code for ECCV2022 submission 3568
This repository contains evaluation code and pre-trained models.

## Setup
* Install required packages. The code has been tested with PyTorch 1.9 / Python 3.8:
   `pip install -r requirements.txt`

* Create folders for datasets and model snapshots:
`mkdir -p checkpoints datasets/street_hazards`

## Datasets
Download and extract the desired dataset:
* StreetHazards:
   * `wget https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar -P datasets/street_hazards/`
   * `tar -xf datasets/street_hazards/streethazards_test.tar -C datasets/street_hazards/`
* RoadAnomaly:
   * `wget https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly_jpg.zip -P datasets/`
   * `unzip datasets/RoadAnomaly_jpg.zip -d datasets/`

## Model Checkpoints
Download the desired checkpoint and place it in the `checkpoints` folder:
* StreetHazards: https://drive.google.com/file/d/1mdTDlOZTjWf1YAIHYiuUFCGKgC4N7KTC/view?usp=sharing
* RoadAnomaly: https://drive.google.com/file/d/18NbxWfKnxpRsyB9CySGFB7-pXjLksy5y/view?usp=sharing

## Dataset Evlauation
Note: since the results reported in the paper are averages over multiple runs, the outcomes obtained with this code will differ slightly.  
* StreetHazards: `python eval.py street_hazards checkpoints/StreetHazards_deeplabv3_resnet50.pth --arch deeplabv3_resnet50`
* RoadAnomaly: `python eval.py road_anomaly checkpoints/BDD100k_deeplabv3plus_resnet101.pth --arch deeplabv3plus_resnet101`
