import gradio as gr
import joblib
import numpy as np
import os
import sklearn

# Define the base directory for models
MODELS_DIR = r"/Users/sohan/Documents/GitHub/email_spam_classifier/models"

# Load vectorizer and models
try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
    models = {
         "Ensemble Model": joblib.load(os.path.join(MODELS_DIR, "ensemble_model.joblib")),
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest_model.joblib")),
        "LightGBM": joblib.load(os.path.join(MODELS_DIR, "lightgbm_model.joblib")),
        "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib")),
        "SVM": joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib")),
        "Naive Bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes_model.joblib"))
    }
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_message = f"Failed to load models or vectorizer: {str(e)}"

def classify_email(email_text, model_name):
    if not email_text.strip():
        return "⚠️ Please enter an email text to classify.", "", "", ""
    
    try:
        model = models[model_name]
        
        # Preprocess and make prediction
        email_features = vectorizer.transform([email_text]).toarray()
        prediction_proba = model.predict_proba(email_features)[0]
        prediction = model.predict(email_features)[0]
        confidence = np.max(prediction_proba)
        
        # Prepare results
        result = "Spam" if prediction else "Ham"
        result_with_icon = f"🚨 Prediction: {result}" if result == "Spam" else f"✅ Prediction: {result}"
        confidence_score = f"Confidence Score: {confidence:.2%}"
        model_info = f"Model Used: {model_name}"
        
        # Confidence level interpretation
        if confidence < 0.6:
            confidence_message = "⚠️ Low confidence prediction. Consider manual review."
        elif confidence > 0.9:
            confidence_message = "ℹ️ High confidence prediction."
        else:
            confidence_message = ""
            
        return result_with_icon, confidence_score, model_info, confidence_message
    
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}", "", "", ""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Spam Email Classifier") as demo:
        gr.Markdown("# 🕵️ Spam Email Classifier")
        
        if not models_loaded:
            gr.Markdown(f"## ❌ Error: {error_message}")
        else:
            gr.Markdown("## ✅ Models and vectorizer loaded successfully!")
            
            # Model selection
            model_dropdown = gr.Dropdown(
                choices=list(models.keys()),
                value="Best Model",
                label="Select a Classification Model"
            )
            
            # Email input
            email_input = gr.Textbox(
                label="Enter Email Subject and Body:",
                placeholder="Paste the email text here...",
                lines=10
            )
            
            # Classify button
            classify_button = gr.Button("Classify Email")
            
            # Output components
            with gr.Group():
                prediction_output = gr.Markdown()
                confidence_output = gr.Markdown()
                model_output = gr.Markdown()
                message_output = gr.Markdown()
            
            # Set up the click event
            classify_button.click(
                fn=classify_email,
                inputs=[email_input, model_dropdown],
                outputs=[prediction_output, confidence_output, model_output, message_output]
            )
    
    return demo

# Run the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)