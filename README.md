# Detection and Prevention of Vehicle Insurance Claim Fraud

## Project Overview

This project aims to tackle the pervasive issue of vehicle insurance fraud, which causes substantial financial losses for insurance companies and erodes consumer trust. Fraudulent claims vary from staged accidents to exaggerated injuries, complicating the claims process and increasing costs. By leveraging historical vehicle and policy data, our objective is to develop a robust predictive model to accurately detect and prevent fraudulent claims. The implementation of this model is intended to help insurance companies minimize financial losses, enhance the efficiency of claims processing, and maintain fair premium pricing for customers.

## Dataset

### Name: Vehicle Claim Fraud Detection

- **Source**: Kaggle
- **Size**: 15,420 records
- **Variables**: 33 (both categorical and numerical)
- **Key Features**:
  - Month of the accident
  - Day of the week
  - Make of the vehicle
  - Accident area
  - Age of the policyholder
  - Various policy details
  - Indicator of whether the claim was fraudulent

The dataset offers a robust sample size for training and evaluating the predictive model and includes indicators for fraudulent claims, making it suitable for building a classification model.

## Project Structure

- <b>Data Preparation:</b> Handle missing values, convert data types, encode categorical variables, and scale numerical features.
- <b>Feature Engineering:</b> Select relevant features, encode categorical variables using one-hot encoding, and apply SMOTE to handle class imbalance.
- <b>Model Training and Evaluation:</b> Train and evaluate models (Isolation Forest, Gradient Boosting, Decision Tree, XGBoost, Random Forest, K-Nearest Neighbor, Logistic Regression, and CatBoost) with hyperparameter tuning.
- <b>Model Comparison:</b> Compare models based on performance metrics, highlighting CatBoost as the top performer.

## Installation and Usage

### Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn
  - scikit-optimize (for Bayesian Optimization)
  - Jupyter Notebook

### Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/oxayavongsa/aai-510-ml-group-1
    cd aai-510-ml-group-1
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook for Exploratory Data Analysis (EDA):
    ```bash
    jupyter notebook Final Project SectionA-Team 1.ipynb
    ```

4. Follow the notebook steps to perform data cleaning, feature selection, and model training.

## Team Members

- **Team Leader/Representative**: Outhai Xayavongsa (Thai)
- **Technical Lead**: Aaron Ramirez
- **Members**:
  - Aaron Ramirez
  - Muhammad Haris
  - Outhai Xayavongsa (Thai)

## YouTube: <a href="https://youtu.be/TztlKFz5VXU?si=MeweLXnsnQG7GRCP" target="_blank">Related Video</a>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Orginal Link: https://github.com/oxayavongsa/aai-510-ml-group-1
