# Email Spam Detection System

![Streamlit](https://img.shields.io/badge/Streamlit-App-blue.svg) ![Python](https://img.shields.io/badge/Python-3.8+-green.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Project Overview
This repository contains an *Email Spam Detection System* that classifies emails as *Spam* or *Not Spam (Ham)* using machine learning. The system is trained on the *Enron Spam Dataset, processes emails through advanced text preprocessing, and deploys multiple models, including an ensemble approach for higher accuracy. A **Streamlit Web App* is also included for user-friendly email classification.

## Key Features
- *Dataset*: Uses the Enron spam dataset with ~33,110 emails (16,542 ham, 16,568 spam).
- *Models*: Trains 5 models (Naive Bayes, XGBoost, Random Forest, LightGBM, SVM) and a weighted voting ensemble.
- *Preprocessing*: Cleans data, removes NaN values, merges 'Subject' and 'Message', applies TF-IDF vectorization (10,000 features).
- *Evaluation*: Calculates accuracy, F1 score, precision, recall, and confusion matrices.
- *Deployment*: Streamlit app for real-time spam classification with model selection and confidence scores.
- *Example Tests*: Provides sample spam and ham emails for user testing.


## Folder Structure

email-spam-detection/
┣ enron_spam_data.csv       # Raw dataset
┣ train_model.py            # Training script
┣ evaluate_model.py         # Model evaluation script
┣ app.py                    # Main Streamlit app script
┣ requirements.txt          # Dependencies list
┣ README.md                 # Project documentation



## 📌 Installation Guide

### 🔧 Prerequisites
Ensure you have the following installed:
- Python *3.8+*
- Git (for cloning the repository)
- At least *20GB RAM* (for training on dense arrays)

### 🔽 Clone the Repository
bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection


### 📦 Install Dependencies
bash
pip install -r requirements.txt


### ⚙ Train the Model
To train the models from scratch, run:
bash
python train_model.py

This will train and save models in the models_improved/ folder.

### 🏆 Evaluate the Model
After training, evaluate the models using:
bash
python evaluate_model.py


### 🌐 Run the Streamlit Web App
To launch the web application for testing:
bash
streamlit run app.py


---

## 📊 Model Performance
| Model          | Accuracy | F1 Score | Precision | Recall  |
|---------------|----------|----------|----------|----------|
| Naive Bayes    | 98.47%   | 98.48%   | 98.22%   | 98.73%   |
| XGBoost        | 96.12%   | 96.24%   | 93.36%   | 99.30%   |
| Random Forest  | 91.33%   | 92.01%   | 85.42%   | 99.69%   |
| LightGBM       | 96.13%   | 96.25%   | 93.43%   | 99.25%   |
| *Ensemble*   | *98.92%* | *98.94%* | *98.75%* | *99.13%* |

---

## 🎭 Example Test Emails
Try these examples in the *Streamlit app*:
#### ✅ Ham Email

Subject: Project Update Request
Hi Team,
Can you please provide me with the latest project update? Let me know if you need any details from my end.
Best regards,
John Doe

#### 🚫 Spam Email

Subject: Congratulations! You Won $1,000,000!!!
Dear User,
You have been selected for a huge cash prize! Click the link below to claim your winnings NOW!
http://spamlink.com


## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.
> **Important**: PR sucks!!!.

## 🛡 License
This project is licensed under the MIT License.

## 📧 Contact
For queries, email *ashishmahendran04@gmail.com* or create an issue in this repository.


⭐ *Star this repo if you find it useful!* 🌟

