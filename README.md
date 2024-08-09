# Music Genre and Composer Classification Using Deep Learning

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) ![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg) 

` Authors: Zain Ali, Angel Benitez, and Outhai Xayavongsa`

## Project Overview

This project focuses on classifying music by genre and composer using deep learning techniques. The primary goal is to accurately identify the composer of a given musical piece by leveraging two deep learning models: Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN). 

Classifying composers based on musical scores is a challenging task due to the nuanced differences in styles and compositions. However, by applying state-of-the-art deep learning models, this project aims to achieve high accuracy in composer classification, contributing to the fields of musicology and artificial intelligence.

## Repository Structure

| File/Folder Name                                        | Description                                                                                           |
|---------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `Music Genre and Composer Classification Using Deep Learning.ipynb` | The Jupyter Notebook containing the entire codebase for the project.                        |
| `Music Genre and Composer Classification Using Deep Learning.pdf`  | A PDF version of the Jupyter Notebook for easy sharing and reviewing.                         |
| `Music Genre and Composer Classification Using Deep Learning.py`   | A Python script version of the Jupyter Notebook for running the code outside of a notebook environment.|
| `raw_data/`                                             | Contains the raw dataset used in the project, including the `midi_classic_music_data.zip` file.        |
| `README.md`                                             | This document.                                                                                         |
| `requirements.txt`                                      | Lists all the dependencies and libraries required to run the project.                                  |
| `LICENSE`                                               | The licensing information for the project.                                                            |
| `.gitignore`                                            | Specifies files and directories that should be ignored by Git.                                         |

## Dataset

The dataset used for this project consists of 3,929 MIDI files of classical works by 175 composers, including Bach, Beethoven, Chopin, and Mozart. The dataset is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music).

For the purpose of this project, the focus is narrowed down to the following four composers:

1. Johann Sebastian Bach
2. Ludwig van Beethoven
3. Frédéric Chopin
4. Wolfgang Amadeus Mozart

These composers were chosen due to their distinct styles and significant contributions to classical music.

**Note:** The `raw_data/` directory already contains the necessary data for the project, so there's no need to download or prepare the dataset unless there are specific changes or updates required.

### All Artists Inclusive Analysis

In addition to the primary analysis focusing on the four composers mentioned above, the project also includes a comprehensive model that analyzes compositions from a broader set of 147 classical composers. This model expands the classification task to cover a wider range of styles and compositional techniques, offering insights into the ability of deep learning models to generalize across a diverse set of musical works.

This inclusive analysis is crucial for understanding the limitations and strengths of the models when applied to a larger and more varied dataset. The findings from this analysis provide a deeper understanding of how well the models can differentiate between composers with varying levels of similarity in their musical styles.

## Methodology

### 1. Data Collection
The MIDI files are downloaded from Kaggle and stored in the `raw_data/` directory. 

### 2. Data Pre-processing
The MIDI files are pre-processed to convert the musical scores into a format suitable for deep learning models. This involves:
- Parsing the MIDI files to extract relevant musical features.
- Applying data augmentation techniques to create a more diverse training set.

### 3. Feature Extraction
Features such as notes, chords, and tempo are extracted using specialized music analysis tools. These features are essential for distinguishing between different composers and genres.

### 4. Model Building
Two deep learning models are developed:
- **LSTM (Long Short-Term Memory):** LSTM networks are used due to their effectiveness in handling sequential data, such as musical scores.
- **CNN (Convolutional Neural Network):** CNNs are applied to capture the spatial relationships within the musical features.

### 5. Model Training
The models are trained using the pre-processed and feature-extracted data. Various hyperparameters are tuned to optimize model performance.

### 6. Model Evaluation
The models are evaluated based on:
- **Accuracy:** The percentage of correctly classified composers.
- **Validation Loss:** The loss function value calculated on the validation dataset during training.

### 7. Model Optimization
Further optimization is performed by fine-tuning the hyperparameters, adjusting the learning rate, batch size, and model architecture.

## Results

### Four Composer Analysis
The results of the model training and evaluation for the four composers (Bach, Beethoven, Chopin, and Mozart) will be included in the project report. This section highlights the accuracy, validation loss, and any insights gained during the project.

### All Artists Inclusive Analysis
The broader model that includes 147 composers provides a more extensive evaluation, revealing how the models perform across a larger and more diverse dataset. The results from this analysis are crucial for understanding the generalization capabilities of the models.

## Usage

### Running the Code
To run the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/zainnobody/AAI-511-Final-Project
   ```
2. Navigate to the project directory:
   ```bash
   cd AAI-511-Final-Project
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Music Genre and Composer Classification Using Deep Learning.ipynb"
   ```
5. Alternatively, run the Python script:
   ```bash
   python "Music Genre and Composer Classification Using Deep Learning.py"
   ```

### Dataset Preparation
The dataset is already prepared and included in the `raw_data/` directory. If new data is introduced or updates are required, follow these steps:
- Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music).
- Unzip the files and place them in the `raw_data/` directory.

### Model Training
Ensure that your environment is set up with the necessary libraries (as specified in the `requirements.txt`). Train the model using the provided data and code.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- The dataset is sourced from Kaggle.
- Special thanks to the course instructor and teammates for their contributions and support throughout the project.

## Contact

For any inquiries or contributions, please contact us within Slack.
