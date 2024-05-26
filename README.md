# Predicting Stroke Risk: A Machine Learning Approach to Early Detection and Prevention
### Author: Outhai Xayavongsa

### Dataset
The Stroke Prediction Dataset, authored by FEDESORIANO, comprises 11 clinical features aimed at predicting stroke events. 

#### This dataset includes attributes such as:
Gender
Age
Hypertension status
Heart disease status
Marital status
Type of work
Residence type
Average glucose level
Body mass index (BMI)
Smoking status
Whether the patient has had a stroke
The data is used to analyze and predict the likelihood of a stroke based on these factors, providing valuable insights for healthcare professionals.

## Problem Statement
The objective of this project is to predict stroke occurrences using the "healthcare-dataset-stroke-data.csv" dataset. Stroke is a leading cause of death and disability worldwide, and early prediction can significantly improve patient outcomes. We aim to develop predictive models using Decision Tree and Random Forest classifiers to identify individuals at high risk of stroke based on various health and demographic features. The project involves handling data imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and evaluating the models' performance using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

### Requirements
Python 3.8+
Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
Imbalanced-learn
Joblib

### Usage
Load and preprocess the data
Perform exploratory data analysis (EDA)
Handle data imbalance with SMOTE
Train and evaluate Decision Tree and Random Forest classifiers
Visualize model performance and feature importance

### Preprocessing
The preprocessing steps include:
Imputing missing values
Encoding categorical variables
Scaling numerical features
Handling class imbalance using SMOTE
Models
We use Decision Tree and Random Forest classifiers to predict stroke occurrences.

### Evaluation Metrics
Accuracy
Precision
Recall
F1-score
ROC AUC
Results
Random Forest Classifier: Higher accuracy (92%) and reasonable ROC AUC score (0.77)
Decision Tree Classifier: Lower accuracy (71%) and ROC AUC score (0.77)
Feature Importance: Age, BMI, and average glucose level are the top three significant features.

### Conclusion
Early prediction of stroke can significantly improve patient outcomes by enabling timely interventions and personalized healthcare strategies. This project underscores the importance of comprehensive feature analysis and the application of robust machine learning models in predicting health outcomes. The insights gained can inform healthcare professionals and policymakers in designing targeted preventive measures and improving patient care.

### Future Work
Incorporate additional health metrics and longitudinal data
Explore more sophisticated models to enhance prediction accuracy

### License
This project is licensed under the MIT License.

### Acknowledgments
Thanks to FEDESORIANO for the Stroke Prediction Dataset.
