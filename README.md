# Geographical Feature Classification - Machine Learning Challenge

**Team Name:** CSMSLM
**Team Members:** David Ngwe Pouth, Omar Amrane El Idrissi, Adnan Slimane Ali

## Project Overview
This repository contains the code used to generate our submission for the Geographical Feature Classification Kaggle competition. We implemented an exhaustive feature engineering pipeline and an Ensemble Stacking model (LightGBM, XGBoost, RandomForest) to predict the geographical classes. We also included a script demonstrating spatial data leakage in the dataset.

## Files Included
* `advanced_benchmark.py`: The main script containing the feature engineering, model training, cross-validation, and submission generation.
* `data_leakage.py`: A script demonstrating the spatial data leakage using geographic coordinates.
* `requirements.txt`: The list of Python dependencies required to run the code.
* `README.md`: This instruction file.

## Prerequisites
Ensure you have Python 3.9+ installed. Install the required libraries using the following command:
```Bash
pip install -r requirements.txt
```

## How to Reproduce the Submission
1. Download the dataset files (`train.geojson` and `test.geojson`) from the Kaggle competition page.
2. Place these two `.geojson` files in the exact same directory as the python scripts.
3. Run the main execution script:
```Bash 
python advanced_benchmark.py
```
4. The script will output execution logs to the console and to a file named `benchmark_execution.log`.
5. Once finished, the final prediction file `stacked_ensemble_submission.csv` will be generated in the same directory, ready to be uploaded to Kaggle.