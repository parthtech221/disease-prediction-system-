# Disease Prediction System

A Machine Learning project that predicts diseases based on symptoms using a Random Forest Classifier. The model is trained on a disease-symptom dataset and can classify diseases from user-provided symptom data.

## Project Overview

This project uses Machine Learning techniques to analyze symptom patterns and predict possible diseases. The trained model and label encoder are stored as serialized files for easy deployment and reuse.

## Features

- Disease prediction based on symptoms
- Random Forest Classification Model
- Label Encoding for disease labels
- Pre-trained model included
- High prediction accuracy
- Easy model loading using Pickle

## Tech Stack

- Python
- Pandas
- Scikit-Learn
- Pickle

## Project Structure

```
Disease-Prediction-System/
│
├── dataset.csv              # Training dataset
├── model.pkl                # Trained Random Forest model
├── label_encoder.pkl        # Label encoder
├── model_accuracy.pkl       # Saved model accuracy
├── load.py                  # Dataset loading script
└── README.md                # Project documentation
```

## Dataset Information

The dataset contains:

- Symptoms as input features
- Disease names as target labels

Dataset loading example:

```python
import pandas as pd

df = pd.read_csv("dataset/disease_symptom.csv")

print(df.head())
print(df.columns)
print(df.shape)
```

## Model Details

| Parameter | Value |
|------------|---------|
| Algorithm | Random Forest Classifier |
| Number of Trees | 50 |
| Max Depth | 12 |
| Random State | 42 |

## Model Accuracy

The saved model achieved:

**Accuracy: 100% (1.0)**

> Note: Accuracy may vary depending on the dataset split and evaluation strategy.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Disease-Prediction-System.git
cd Disease-Prediction-System
```

Install dependencies:

```bash
pip install pandas scikit-learn numpy
```

## Loading the Trained Model

```python
import pickle

model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
accuracy = pickle.load(open("model_accuracy.pkl", "rb"))

print("Model Accuracy:", accuracy)
```

## Making Predictions

```python
prediction = model.predict([input_features])

disease = label_encoder.inverse_transform(prediction)

print("Predicted Disease:", disease[0])
```

## Future Improvements

- Web-based interface using Flask or Streamlit
- Real-time symptom input
- Disease probability scores
- Medical recommendation module
- Deployment on cloud platforms

## Disclaimer

This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Author

Parth Patel

Machine Learning & AI Enthusiast
