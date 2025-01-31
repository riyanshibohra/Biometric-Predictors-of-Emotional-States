<p align="center">
  <h1>Biometric Predictors of Emotional States: A Deep Learning Approach 🧠</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
  <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
  <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=for-the-badge">
</p>

This project explores the fascinating intersection of artificial intelligence and human emotions. By leveraging wearable technology data, I delve into how biometric indicators can be used to understand and predict emotional states. The project employs advanced machine learning techniques, including deep learning and transformer models, to analyze both synthetic and real-world datasets.

## Project Overview

### Part 1: Mood State Generation and Classification
A synthetic dataset is used to generate mood descriptions using a pre-trained GPT-2 model. These descriptions are then classified into predefined mood states using various deep learning models, including an Ensemble model and a BERT model.

### Part 2: Stress Level Classification from Real-World Data
Using real-world data, stress levels are classified into distinct categories. This part of the project involves the application of classification algorithms such as Decision Trees, Random Forest, Logistic Regression, and Gradient Boosting.

## Methodology
- **Data Processing:** Includes loading, cleaning, and preprocessing data from wearables.
- **Model Training:** Involves training deep learning models for text generation and mood classification.
- **Evaluation:** Models are evaluated based on metrics like precision, recall, and F1 score, providing insights into their performance in identifying emotional states.

## Results
The findings highlight the superior performance of the Ensemble model in classifying mood states in synthetic data, while the Random Forest model excelled in handling real-world data for stress level classification.

## Project Directory

### `code/`
- `Description.md`: Documentation and descriptions of the code structure.
- `Part1_Mood_descriptions.py`: Script for generating mood descriptions using GPT-2.
- `Part2_Stress_Level.py`: Script for classifying stress levels from real-world data.
- `generate_descriptions.py`: Utility script for generating descriptive text based on data inputs.

### `data/`
- `README.md`: Detailed information about the datasets used.
- `Sleep_health_and_lifestyle_dataset.csv`: Real-world dataset including various health metrics.
- `activity_environment_data.csv`: Data on activity and environmental factors.
- `digital_interaction_data.csv`: Dataset capturing digital interactions.
- `mood_descriptions.csv`: Synthetic dataset with mood descriptions.
- `personal_health_data.csv`: Additional health metrics and personal data.

### `figures/`
- **Real-life Dataset:**
  - `f1score_comparison.png`: Comparison of F1 scores across models.
  - `precision_comparison.png`: Comparison of precision across models.
  - `recall_comparison.png`: Comparison of recall across models.
- **Synthetic Data Results:**
  - `f1_plot.png`: F1 score visualization for synthetic data models.
  - `precision_plot.png`: Precision visualization for synthetic data models.
  - `recall_plot.png`: Recall visualization for synthetic data models.

### `Report.pdf`
- A comprehensive report detailing the project's objectives, methodologies, results, and conclusions.

## Conclusion
This project showcases the potential of integrating AI with biometric data to advance our understanding of human emotions, offering promising applications in mental health monitoring and intervention.
