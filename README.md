# Source code for "Probing Contextual Diversity for Dense Out-of-Distribution Detection"
This repository contains the evaluation code and some pre-trained models.

## How to use
1. Clone the repository and move to its directory

2. Create folders for datasets and model snapshots:
`mkdir -p checkpoints datasets/street_hazards`

3. Download and extract the desired dataset:
    * StreetHazards:
        * `wget https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar -P datasets/street_hazards/`
        * `tar -xf datasets/street_hazards/streethazards_test.tar -C datasets/street_hazards/`
    * RoadAnomaly: 
        * `wget https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly_jpg.zip -P datasets/`
        * `unzip datasets/RoadAnomaly_jpg.zip -d datasets/`

4. Download the desired checkpoint and place it in the `checkpoints` folder:
    * StreetHazards: https://drive.google.com/file/d/1mdTDlOZTjWf1YAIHYiuUFCGKgC4N7KTC/view?usp=sharing
    * RoadAnomaly: https://drive.google.com/file/d/1O5FNijvCvqUybOweAnsHPTXkiAcTQmKa/view?usp=sharing

5. Run the evaluation:
    * StreetHazards: `python eval.py street_hazards checkpoints/StreetHazards_deeplabv3_resnet50.pth --arch deeplabv3_resnet50`
    * RoadAnomaly: `python eval.py road_anomaly checkpoints/BDD100k_deeplabv3plus_resnet101.pth --arch deeplabv3plus_resnet101`
    