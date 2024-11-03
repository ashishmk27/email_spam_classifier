import streamlit as st
from joblib import load

# Load the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = load('spam_classifier.joblib')
    vectorizer = load('tfidf_vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Streamlit app title and description
st.title("Email Spam Classifier")
st.write("Enter an email message: spam or ham?")

# User input
user_input = st.text_area("Message", "Type your email message here...")

# Prediction
if st.button("Predict"):
    if user_input:
        # Vectorize the input message
        input_vector = vectorizer.transform([user_input])
        
        # Predict and display the result
        prediction = model.predict(input_vector)[0]
        label = "Spam" if prediction == 1 else "Ham"
        st.write(f"Prediction: **{label}** ({prediction})")
    else:
        st.write("Please enter a message to classify.")