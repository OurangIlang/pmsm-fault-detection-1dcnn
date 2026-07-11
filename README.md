# PMSM Predictive Maintenance Dashboard

This repository contains a 1D Convolutional Neural Network (1D-CNN) model and a Streamlit web application for detecting stator faults in a 1.0 kW Permanent Magnet Synchronous Motor (PMSM). The system analyzes Z-axis accelerometer vibration data to classify the motor's condition into three categories: `normal`, `coil_short`, and `interturn_short`.

## Dataset
The model is trained on a public vibration dataset. The dataset is excluded from this repository to comply with file size constraints. 
* **Source**: [Mendeley Data - PMSM Vibration Dataset](https://data.mendeley.com/datasets/rgn5brrgrn/5)
* **Limitations**: The pre-trained weights (`pmsm_vibration_model.h5`) are only validated for 1.0 kW motor operations. Uploading `.tdms` files from 1.5 kW or 3.0 kW motors to the dashboard will yield inaccurate classifications.
* **Instructions**: Download the `.tdms` files from the provided URL. To execute the training pipeline, extract the files into your designated local data directory and update the path in the Jupyter Notebook.

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
