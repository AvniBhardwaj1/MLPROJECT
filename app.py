class BoostedCosineClassifier:
    def __init__(self, base_model=None):
        self.base_model = base_model

    def fit(self, X, y):
        return self.base_model.fit(X, y)

    def predict(self, X):
        return self.base_model.predict(X)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import os

# === Load Avni’s saved models ===
model_dir = r"C:\Users\heman\OneDrive\Documents\VI SEM\ML\MLPROJECT\models"

with open(os.path.join(model_dir, "cosine_enhanced_model.pkl"), "rb") as f:
    cosine_model_data = pickle.load(f)

with open(os.path.join(model_dir, "news_detector.pkl"), "rb") as f:
    simple_model_data = pickle.load(f)

# Extract components
tfidf = cosine_model_data['tfidf']
cosine_model = cosine_model_data['model']
train_vectors = cosine_model_data['train_vectors']
train_labels = cosine_model_data['train_labels']

simple_model = simple_model_data['cosine_model']

# === Streamlit UI ===
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

# Optional background styling
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
    }
    .main {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detection App")
st.subheader("Check whether a news article is **Real** or **Fake** using ML & NLP")
st.markdown("---")

# === Example articles ===
example_articles = {
    "🗞️ COVID-19 vaccine proven 95% effective, WHO confirms": 
        "The World Health Organization has confirmed that the newly developed COVID-19 vaccine shows 95% effectiveness.",
    "👽 Alien spaceship spotted over Mumbai, scientists confirm!":
        "Multiple reports from Mumbai suggest an alien spaceship was spotted last night. Scientists say signals are extraterrestrial.",
    "🏛️ Govt to launch new startup scheme for students": 
        "The Indian Government has proposed a new startup support scheme for college students and recent graduates.",
}

example_choice = st.selectbox("💡 Try an example article:", [""] + list(example_articles.keys()))

user_input = st.text_area("✍️ Enter news text to analyze:", value=example_articles.get(example_choice, ""))

# === Prediction ===
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text or select an example.")
    else:
        # Vectorize input
        input_vector = tfidf.transform([user_input])

        # Compute cosine similarity
        cosine_sim = cosine_similarity(input_vector, train_vectors)[0]
        real_sim = np.mean(cosine_sim[train_labels == 1])
        fake_sim = np.mean(cosine_sim[train_labels == 0])
        cosine_features = np.array([[real_sim, fake_sim]])

        # Combine features
        combined_input = hstack([input_vector, cosine_features])

        # Predict using cosine-enhanced model
        prediction = cosine_model.predict(combined_input)[0]
        confidence = np.max(cosine_model.predict_proba(combined_input))

        st.markdown("## 📊 Prediction Result")
        if prediction == 1:
            st.success("✅ This looks like **REAL** news.")
        else:
            st.error("❌ This seems to be **FAKE** news.")

        st.write(f"**Confidence Score:** `{confidence:.2f}`")
        st.write(f"**Similarity with Real News:** `{real_sim:.2f}`")
        st.write(f"**Similarity with Fake News:** `{fake_sim:.2f}`")

st.markdown("---")
st.caption("🤖 Developed by Arushi Dubey" \
" Arohi Agrawal " \
"\n Avni Bharadwaj" \
" \n Hemanshi Balani" \
" \n Himadri Rathore")