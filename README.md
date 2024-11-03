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


# Spam Classifier

This repository contains a machine learning-based spam classifier that categorizes emails as **Spam** or **Not Spam**. The model is trained using natural language processing techniques with a TF-IDF vectorizer.

## Requirements

To use the spam classifier, install the required libraries:

```bash
pip install numpy pandas scikit-learn xgboost joblib
```

## Usage

Follow the steps below to classify new emails.

### Step 1: Load the Pre-trained Model

Load the saved spam classifier model from the `spam_classifier.joblib` file:

```python
from joblib import load

# Load the saved model
model = load('spam_classifier.joblib')
```

### Step 2: Load the TF-IDF Vectorizer

Since the model was trained with a TF-IDF vectorizer, you’ll need to load the saved vectorizer as well:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved TF-IDF vectorizer
vectorizer = load('tfidf_vectorizer.joblib')
```

### Step 3: Vectorize the New Email

Prepare and transform the new email text using the loaded vectorizer:

```python
# Example email
new_email = "Congratulations! You've won a free prize. Click here to claim now!"

# Transform the email text to match the vectorizer format
new_email_vectorized = vectorizer.transform([new_email])
```

### Step 4: Predict the Email Label

Use the model to predict if the email is **Spam** or **Not Spam**:

```python
# Make a prediction
prediction = model.predict(new_email_vectorized)

# Print the result
if prediction[0] == 0:
    print("Not Spam")
else:
    print("Spam")
```

### Example Output

For the above example, the output might be:

```
Spam
```

## Repository Files

- `spam_classifier.joblib`: The pre-trained spam classifier model.
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer used to process the email text.
- `README.md`: Instructions for using the spam classifier.

## Notes

- Ensure that both `spam_classifier.joblib` and `tfidf_vectorizer.joblib` files are located in the same directory as your script.
- The classifier may not generalize well to all types of emails, as it was trained on specific data.

---

This README file is now formatted and ready for GitHub, with clear headers, code blocks, and explanations. This format will be easy for users to read and follow.


 ## Conclusion

This project demonstrates the effectiveness of machine learning in spam email classification. By leveraging different algorithms and evaluating their performance, we can build a robust system for identifying and filtering spam emails. Future improvements could involve exploring more advanced techniques and incorporating larger datasets for enhanced accuracy.
