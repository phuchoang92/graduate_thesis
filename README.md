# Fisheye Video Processing and Action Detection Pipeline

This repository provides scripts and instructions for fisheye distortion correction, action detection pipeline, and a demo web app that combines both functionalities.

## Requirements

To install the required Python packages, run:
```sh
pip install -r requirements.txt
```

## For fisheye distortion correction only

```sh
cd fisheye
python unfish.py <input_video>
```

## For running pipeline

```sh
python -d ava_v2.2 -v yowo_v2_large -size 224 --weight <checkpoints> --video <input_video>
```

## For demo web app

Have action detection + fisheye undistortion


| Streamlit        |
|-------------------------| 
| `streamlit run webapp.py` |


## Checkpoints weight link

To get all model checkpoint weights, use this [google drive link](https://drive.google.com/drive/folders/1RUF1I7gAgMsfmC_r-dnWh2eam_HG5ELP?usp=sharing)

## Dataset link

All dataset is stored in this [google drive link](https://drive.google.com/drive/folders/1y_pVVUCDIS2TUUg5HtRzMJw-ZrVT7OOZ?usp=sharing)

