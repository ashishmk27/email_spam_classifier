import streamlit as st
import joblib
import numpy as np
import os

# Define the base directory for models
MODELS_DIR = r"C:\Users\Ashish Mahendran\New folder (16)\models_improved"

# Streamlit app configuration (must be the first Streamlit command)
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")

# Load vectorizer and models with error handling
try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
    models = {
        "Best Model": joblib.load(os.path.join(MODELS_DIR, "best_model.joblib")),
        "Ensemble Model": joblib.load(os.path.join(MODELS_DIR, "ensemble_model.joblib")),
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib")),
        "LightGBM": joblib.load(os.path.join(MODELS_DIR, "lightgbm_model.joblib")),  # Fixed typo "Ligtgm" to "LightGBM"
        "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib")),
        "SVM": joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib")),
        "Naive Bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes_model.joblib"))
    }
except Exception as e:
    st.error(f"Failed to load models or vectorizer: {str(e)}")
    st.stop()

# Main app
def main():
    st.title("🕵️ Spam Email Classifier")
    st.success("Models and vectorizer loaded successfully!")  # Moved inside main()
    
    # Model selection
    model_name = st.selectbox("Select a Classification Model", list(models.keys()))
    model = models[model_name]
    
    # Email input
    email_text = st.text_area("Enter Email Subject and Body:", 
                              placeholder="Paste the email text here...", 
                              height=250)
    
    # Classify button
    if st.button("Classify Email"):
        if email_text.strip():
            try:
                # Preprocess: Convert sparse matrix to dense array to match training data
                email_features = vectorizer.transform([email_text]).toarray()
                prediction_proba = model.predict_proba(email_features)[0]
                prediction = model.predict(email_features)[0]
                confidence = np.max(prediction_proba)
                
                # Display results
                result = "Spam" if prediction else "Ham"
                
                # Result display with color coding
                if result == "Spam":
                    st.error(f"### 🚨 Prediction: {result}")
                else:
                    st.success(f"### ✅ Prediction: {result}")
                
                # Confidence score visualization
                st.write(f"### Confidence Score: {confidence:.2%}")
                
                # Additional model information
                st.write(f"### Model Used: {model_name}")
                
                # Confidence level interpretation
                if confidence < 0.6:
                    st.warning("Low confidence prediction. Consider manual review.")
                elif confidence > 0.9:
                    st.info("High confidence prediction.")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            st.warning("Please enter an email text to classify.")

# Run the app
if __name__ == "__main__":
    main()