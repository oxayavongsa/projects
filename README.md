# Context-Aware Movie Chatbot: Multi-Turn Conversations with Sentiment-Driven Responses

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-green.svg)](https://github.com/oxayavongsa/NLP-Chatbot)
[![GitHub issues](https://img.shields.io/github/issues/oxayavongsa/NLP-Chatbot.svg)](https://github.com/oxayavongsa/NLP-Chatbot/issues)
[![GitHub stars](https://img.shields.io/github/stars/oxayavongsa/NLP-Chatbot.svg)](https://github.com/oxayavongsa/NLP-Chatbot/stargazers)
[![GitHub license](https://img.shields.io/github/license/oxayavongsa/NLP-Chatbot.svg)](https://github.com/oxayavongsa/NLP-Chatbot/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)

## Project Overview
This project is part of the AAI-520: Natural Language Processing course in the **Applied Artificial Intelligence Program** at the University of San Diego under the guidance of **Professor Kahila Mokhtari, Ph.D.** Our goal is to design and implement a **generative-based chatbot** that not only engages in multi-turn conversations but also incorporates **sentiment analysis** to adapt its responses based on the emotional tone of user input.

The chatbot is trained using the **Cornell Movie Dialogs Corpus**, enabling it to handle diverse conversations with coherence, context-awareness, and emotional sensitivity.

[Visit the Cornell Movie-Dialogs Corpus](https://www.kaggle.com/datasets/rajathmc/cornell-moviedialog-corpus)

# Repository Structure (edit when project complete)


| File/Folder Name         | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| `Final Project Report-Team 6.ipynb` | The Jupyter Notebook containing all the code for building and evaluating the chatbot. |
| `Final Project Deliveries/` | Final Project Code, PowerPoint Presentation, and Brief Report |
| `requirements.txt`        | Dependencies required for the project (e.g., TensorFlow, PyTorch, Hugging Face). |
| `README.md`              | The project overview and structure (this file).                             |
| `data/`                  | Raw dataset files, including the Cornell Movie-Dialog Corpus.               |
| `models/`                | Trained model checkpoints.                                                  |
| `.gitignore`             | Lists files/directories ignored by Git.                                     |
| `LICENSE`                | Licensing information for the project.                                      |

---
## Final Project Report

The final deliverables for this project include a comprehensive report in **PDF format** that contains:

- The full project report
- The Jupyter Notebook code
- Powerpoint presentation
- References and additional materials

### Final Deliverables:
`Final Project Deliveries/`
- **Final_Project_Notebook_Team_6.pdf**: The full notebook converted to PDF, containing all code, analysis, and chatbot implementation.
- **Final_Project_Report_Team_6.pdf**: The final project report, including methodology, results, and references.
- **Final_PowerPoint_Presentation_Team_6.pptx**: The PowerPoint presentation summarizing key points and project progress.

Please refer to these files for all the details regarding the project, methodology, and evaluation.

## Project Status: ✅ Completed

## Team Members:
- **Outhai Xayavongsa** - Team Leader [![GitHub](https://img.shields.io/badge/GitHub-oxayavongsa-lightgrey)](https://github.com/oxayavongsa)
- **Saad Saeed** - Lead Assistant [![GitHub](https://img.shields.io/badge/GitHub-SaadaSaeed86-lightgrey)](https://github.com/SaadaSaeed86)
- **Anand Fernandes** - Team Member [![GitHub](https://img.shields.io/badge/GitHub-af0808-lightgrey)](https://github.com/af0808)

## Installation

To install and run the project on your machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/oxayavongsa/NLP-Chatbot.git
   cd NLP-Chatbot

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv chatbot-env
   source chatbot-env/bin/activate

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

4. **Download the Cornell Movie-Dialog Corpus**:
   ```bash
   kaggle datasets download -d rajathmc/cornell-moviedialog-corpus
   unzip cornell-moviedialog-corpus.zip

5. **Run the Chatbot**:
   ```bash
   python NLP_Chatbot.py

## Run the Chatbot
* Option 1: Open Chatbot.ipynb in Jupyter Notebook or Google Colab to run the chatbot interactively.
* Option 2: Use the command line interface to interact with the chatbot (see step 5 above).

## Dataset Information
* We used the Cornell Movie-Dialogs Corpus which contains:

* 220,579 conversational exchanges between 10,292 pairs of movie characters.
* 9,035 characters from 617 movies.
* In total, 304,713 utterances with metadata such as genres, release year, IMDB rating, and character gender.

## Methods Used
* Python for development.
* PyTorch for model training.
* Hugging Face Transformers for leveraging pre-trained models (T5).
* Jupyter Notebook or Google Colab for experimentation and testing.

## Technologies
* Preprocessing: Data was cleaned by removing punctuation, stopwords, lemmatization, and rare words.
* Model Architecture: T5 (Text-To-Text Transfer Transformer) is used for multi-turn conversations and context-aware responses.
* Sentiment Analysis: Incorporated into the chatbot to adjust responses based on the user's emotional tone.

## How the Chatbot Works
The chatbot uses a **Transformer-based model (T5)** to maintain multi-turn conversations. The **sentiment analysis** layer allows the chatbot to detect and adapt to emotional cues in the user's input, generating appropriate responses. It has been trained using the **Cornell Movie-Dialogs Corpus**, giving it the ability to handle movie-like dialogues with contextual coherence.

### Example Usage
(Replace the above link with an actual gif demo or video)

## Future Improvements
* Enhanced context retention over longer conversations.
* Fine-tuning the model for specific conversational styles or tones.
* Improving the user interface for better interaction.

## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgements
* Thanks to Cristian Danescu-Niculescu-Mizil and Lillian Lee for providing the Cornell Movie-Dialogs Corpus.
* Special thanks to Professor Kahila Mokhtari, Ph.D., for guidance throughout the course.
* Collaboration tools: GitHub, Slack, and Jupyter Notebook/Google Colab.
