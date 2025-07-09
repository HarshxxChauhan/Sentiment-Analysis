import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')

# Load stopwords just in case we use them later
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english'))

# Load model and vectorizer
with open('clf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# THEN test manually
sample = "I love this movie! It was amazing and uplifting."
cleaned = preprocess_text(sample)
vec = vectorizer.transform([cleaned])
print("Manual test (console):", model.predict(vec)[0])

# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Enter a sentence to classify its sentiment:")

user_input = st.text_area("Input Text", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess_text(user_input)
        vector_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector_input)[0]

        # Debug info
        

        if prediction == "pos" or prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😠")
