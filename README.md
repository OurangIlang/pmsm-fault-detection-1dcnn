# PMSM Predictive Maintenance Dashboard

This repository contains a 1D Convolutional Neural Network (1D-CNN) model and a Streamlit web application for detecting stator faults in a 1.0 kW Permanent Magnet Synchronous Motor (PMSM). The system analyzes Z-axis accelerometer vibration data to classify the motor's condition into three categories: `normal`, `coil_short`, and `interturn_short`.

## Dataset
The model is trained on a public vibration dataset. The dataset is excluded from this repository to comply with file size constraints. 
* **Source**: [Mendeley Data - PMSM Vibration Dataset](https://data.mendeley.com/datasets/rgn5brrgrn/5)
* **Limitations**: The pre-trained weights (`pmsm_vibration_model.h5`) are explicitly trained and validated only for 1.0 kW motor operations using vibration data. Uploading .tdms files containing current data, or any data from 1.5 kW and 2.0 kW motor configurations, will result in inaccurate classifications.
* **Instructions**: Download the 1.0 kW .tdms vibration files from the provided Mendeley URL. To execute the training pipeline, extract these files into your local directory and update the DATA_DIR path in the Jupyter Notebook.

## Features
* **1D-CNN Architecture**: Deep learning model configured for time-series vibration signal classification.
* **Streamlit Interface**: A web application (`app.py`) for uploading raw `.tdms` logs and displaying real-time diagnostic results and confidence metrics.
* **Automated Data Extraction**: Parses raw `.tdms` files to extract 2048-point segments for model inference.

## Repository Structure
```text
├── .gitignore
├── app.py                      # Streamlit dashboard application
├── label_classes.npy           # Encoded target labels
├── Laporan SC - Kelompok 8.pdf # Project documentation and report
├── pmsm_vibration_model.h5     # Pre-trained 1D-CNN model weights
├── model.ipynb                 # Model training and preprocessing pipeline
├── README.md
└── requirements.txt            # Environment dependencies
