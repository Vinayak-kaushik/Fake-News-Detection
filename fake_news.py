# ==============================
# Fake News Detection App
# ==============================

import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detection App")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    fake = pd.read_csv(r"C:\Users\Vinayak\Downloads\archive\Fake.csv")
    true = pd.read_csv(r"C:\Users\Vinayak\Downloads\archive\True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

df = load_data()

# ==============================
# VISUALIZATION SECTION
# ==============================

st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.write("### Label Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='label', data=df, ax=ax1)
    ax1.set_xticklabels(['Fake', 'Real'])
    st.pyplot(fig1)

with col2:
    st.write("### Text Length Distribution")
    df['length'] = df['text'].apply(len)
    fig2, ax2 = plt.subplots()
    sns.histplot(df['length'], bins=50, ax=ax2)
    st.pyplot(fig2)

# ==============================
# PREPROCESSING
# ==============================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["text"] = df["text"].apply(clean_text)

# ==============================
# MODEL TRAINING
# ==============================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==============================
# MODEL PERFORMANCE
# ==============================

st.subheader("📈 Model Performance")

col3, col4 = st.columns(2)

with col3:
    st.write("### Accuracy")
    st.write(accuracy_score(y_test, y_pred))

with col4:
    st.write("### Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax3)
    st.pyplot(fig3)

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# ==============================
# USER INPUT SECTION
# ==============================

st.subheader("🧪 Test Your Own News")

user_input = st.text_area("Enter News Article:")

def predict_news(news):
    news = clean_text(news)
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)
    return prediction[0]

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_news(user_input)
        if result == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")