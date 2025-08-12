# Fake News Detector

A simple machine learning project that detects whether a news article or headline is **Fake** or **Real** using a Logistic Regression model trained on the popular Kaggle Fake & Real News dataset.

---

## Project Overview

This project implements a text classification pipeline with the following components:

- **Data:** The Kaggle dataset containing two CSV files: `Fake.csv` and `True.csv`.
- **Preprocessing:**  
  - Combine title and article text.  
  - Clean and lowercase text.  
  - Remove stop words using TF-IDF vectorizer.
- **Model:**  
  - TF-IDF vectorization of text (uni- and bi-grams).  
  - Logistic Regression classifier trained to distinguish fake vs real news.
- **Prediction Interface:**  
  - A Streamlit web app that allows users to:  
    - Paste an article or headline and get a prediction with confidence.  
    - Upload CSV files for batch predictions.
- **Model saving/loading:**  
  - The trained model pipeline is saved using `joblib` to the `models/` directory and loaded by the app.

---

## How to Use

### 1. Prepare the Data

- Download the dataset from Kaggle or use the ones in this repo: [Fake & Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- Place `Fake.csv` and `True.csv` files inside the `data/` directory.

### 2. Train the Model

Run the training script to train the model and save it locally:

```bash
python train.py
