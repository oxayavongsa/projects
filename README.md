Life Expectancy Prediction by US State and Gender
This project aims to predict life expectancy based on the US state and gender using machine learning models. The project involves data cleaning, exploratory data analysis, feature engineering, model training, and evaluation.

Table of Contents
Introduction
Data Description
Installation
Usage
Model Training and Evaluation
Prediction
Results
License
Introduction
This project predicts life expectancy based on US states and gender using datasets from the years 2010-2015 and 2020. The models used include Linear Regression and Random Forest Regressor. The project also includes hyperparameter tuning using GridSearchCV to improve the Random Forest model's performance.

Data Description
Datasets
U.S. Life Expectancy at Birth by State and Census Tract (2010-2015)
Columns: State, County, Census Tract Number, Life Expectancy, Life Expectancy Range, Life Expectancy Standard Error
U.S. State Life Expectancy by Sex (2020)
Columns: State, Sex, LE (Life Expectancy), SE (Standard Error), Quartile
Data Cleaning
Unnecessary columns were dropped.
Missing values were handled by filling with the mean of respective columns.
Categorical variables were encoded for model training.
