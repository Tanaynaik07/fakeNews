# train.py
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATA_DIR = "./Dataset/"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    # Expect Kaggle "Fake.csv" and "True.csv" in data/
    fake = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
    true = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    # combine title + text if both exist
    if "title" in df.columns and "text" in df.columns:
        df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    else:
        # fallback to any text-like column
        txt_cols = [c for c in df.columns if "text" in c or "body" in c or "content" in c]
        df["content"] = df[txt_cols[0]].fillna("") if txt_cols else df.iloc[:,0].astype(str)
    df = df[["content", "label"]].dropna()
    return df

def simple_clean(s):
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return s.strip()

def main():
    df = load_data()
    df["content"] = df["content"].apply(simple_clean)

    X = df["content"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.85, min_df=5)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # evaluation
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1]
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # save pipeline
    model_path = os.path.join(MODEL_DIR, "fake_news_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print("Saved model to", model_path)

if __name__ == "__main__":
    main()
