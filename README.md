# Spam Email Classifier

This project aims to build a robust spam email classifier using machine learning techniques. It leverages various algorithms such as SVM, Naive Bayes, XGBoost, Random Forest, and Gradient Boosting to achieve high accuracy in identifying spam emails.

## Dataset

The project utilizes a subset of a spam email dataset containing 20,000 rows. The 'text' column contains the email content, and the 'label' column indicates whether the email is spam (1) or not spam (0).

## Methodology

1. **Data Preprocessing:**
   - The dataset is loaded using pandas.
   - The data is split into training (80%) and testing (20%) sets.
   - Text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency).

2. **Model Training and Evaluation:**
   - Several classification models are trained and evaluated, including:
     - Support Vector Machine (SVM)
     - Naive Bayes
     - XGBoost
     - Random Forest
     - Gradient Boosting
   - Model performance is assessed using metrics like accuracy, precision, recall, and F1-score.
   - Confusion matrices are generated to visualize the model's predictions.

3. **Model Selection and Deployment:**
   - Based on the evaluation results, the best-performing model is selected for deployment.
   - The selected model (in this case XGBoost is selected) is saved using joblib for future use.

## Results

| Model           | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| SVM             | 0.98325  | 0.97688  | 0.99022 | 0.98351 |
| Naive Bayes     | 0.96975  | 0.99861  | 0.94314 | 0.96998 |
| XGBoost         | 0.987   | 0.9811   | 0.99346 | 0.98724 |
| Random Forest   | 0.985   | 0.97149  | 0.99911 | 0.98511 |
| Gradient Boosting| 0.97825  | 0.96796  | 0.98953 | 0.97863 |


## Usage

To use the spam classifier, follow these steps:

1. Install the required libraries:
pip install numpy pandas scikit-learn xgboost joblib

2. Load the saved model:
from joblib import load model = load('spam_classifier.joblib')
3. Vectorize the new email using the same TF-IDF vectorizer used during training:
python from sklearn.feature_extraction.text import TfidfVectorizer vectorizer = load('tfidf_vectorizer.joblib')
 Load the vectorizer as well, you have saved it in a file
 new_email_vectorized = vectorizer.transform([new_email])
 4. Predict the label for the new email:
 prediction = model.predict(new_email_vectorized) if prediction[0] == 0: print('Not Spam') else: print('Spam')


 ## Conclusion

This project demonstrates the effectiveness of machine learning in spam email classification. By leveraging different algorithms and evaluating their performance, we can build a robust system for identifying and filtering spam emails. Future improvements could involve exploring more advanced techniques and incorporating larger datasets for enhanced accuracy.
