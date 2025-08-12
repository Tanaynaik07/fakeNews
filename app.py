# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/fake_news_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Paste an article or headline and this model will predict *Fake* or *Real* (probabilities included).")

model = load_model()

st.subheader("Single prediction")
text = st.text_area("Enter article text or headline", height=200)
if st.button("Predict"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0]
        label = "Real" if pred == 1 else "Fake"
        conf = prob[pred]
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {conf:.2%}")

        # show top supporting words (if pipeline has tfidf + clf)
        try:
            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            feat_names = tfidf.get_feature_names_out()
            coefs = clf.coef_[0]
            top_pos = np.argsort(coefs)[-10:][::-1]
            top_neg = np.argsort(coefs)[:10]
            st.markdown("**Top features that push toward Real:**")
            st.write(", ".join(feat_names[top_pos]))
            st.markdown("**Top features that push toward Fake:**")
            st.write(", ".join(feat_names[top_neg]))
        except Exception:
            pass

st.subheader("Batch prediction (CSV)")
uploaded = st.file_uploader("Upload CSV with a column named 'content' or 'text' or 'title' for batch prediction", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    # find a text column
    for c in ["content", "text", "title"]:
        if c in df.columns:
            text_col = c
            break
    else:
        text_col = df.columns[0]
    df["content_for_pred"] = df[text_col].astype(str)
    preds = model.predict(df["content_for_pred"].tolist())
    probs = model.predict_proba(df["content_for_pred"].tolist())[:,1]
    df["pred_label"] = np.where(preds==1, "Real", "Fake")
    df["pred_prob_real"] = probs
    st.dataframe(df[[text_col, "pred_label", "pred_prob_real"]].head(50))
    st.markdown("Download full results:")
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")
