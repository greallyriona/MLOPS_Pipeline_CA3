# MLOps Pipeline for Breast Cancer Prediciton Application
### Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Overview
This project implements an end-to-end MLOps pipeline for predicting whether a breast tumour is benign or malignant.

The system uses a machine learning model trained on the Wisconsin Breast Cancer dataset and allows users to input key tumour features through a web interface where they receive real-time predictions.

## Features
- Data preprocessing pipeline
- Logistic Regression model for classification
- Automated workflow using GitHub Actions
- Docker containerisation for deployment
- Flask API for prediction
- Deployed on a virtual machine
- Simple web interface for user input and model output

## Model Inputs
The model uses the following features:
- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean

## Pipeline Overview
1. Data preprocessing
2. Model training
3. Automated workflow (CI/CD)
4. Docker build and deployment
