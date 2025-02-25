Restaurant Review Sentiment Analysis

📌 Project Overview

This project focuses on sentiment analysis of restaurant reviews using Natural Language Processing (NLP). The goal is to classify customer reviews as positive or negative, helping businesses understand customer feedback effectively.

📂 Dataset

The dataset contains restaurant reviews with corresponding sentiment labels:

1: Positive review
0: Negative review

The data is stored in a TSV (Tab-Separated Values) file.

🛠️ Technologies Used

Python
Pandas, NumPy – Data handling and preprocessing
NLTK, re – Text preprocessing
Scikit-learn – Model training and evaluation
Matplotlib, Seaborn – Data visualization

🔄 Data Preprocessing

Text Cleaning: Removed special characters, stopwords, and converted text to lowercase.
Tokenization: Split reviews into words.
Feature Engineering:
Word count, character count
TF-IDF Vectorization

🏆 Model Training & Evaluation

Used multiple models for classification:
Naïve Bayes
Logistic Regression
Random Forest

Evaluated models using accuracy, precision, recall, and F1-score.

📊 Results

The best-performing model achieved high accuracy, successfully distinguishing between positive and negative reviews.
