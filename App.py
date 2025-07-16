import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📩 Spam Message Detector - AI Tool")
st.markdown("Enter a message below to check whether it's **Spam** or **Safe**.")

# Load dataset & train model
@st.cache_data
def train_model():
    data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/spam.csv")
    data = data[['label', 'text']]
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    X = data['text']
    y = data['label']

    cv = CountVectorizer()
    X = cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, cv

model, cv = train_model()

# User Input
user_input = st.text_area("✍️ Enter your message here:")

if st.button("🔍 Check"):
    vect = cv.transform([user_input])
    result = model.predict(vect)
    if result[0] == 1:
        st.error("🚨 Spam Message Detected!")
    else:
        st.success("✅ This message is safe.")
