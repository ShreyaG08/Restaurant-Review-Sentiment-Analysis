import streamlit as st
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\Dell\Downloads\archive (13)\Restaurant_Reviews.tsv', sep='\t')

# Define custom stopwords and remove them from the standard stopwords list
my_stopwords = {"don","don't","ain","aren","aren't","couldn","couldn't","didn","didn't","doesn","doesn't","hadn","hadn't",
               "hasn","hasn't","haven","haven't","isn","isn't","ma","mightn","mightn't","mustn","mustn't","needn","needn't",
               "shan","shan't","no","nor","not","shouldn","shouldn't","wasn","wasn't","weren","weren't","won","won't","wouldn",
               "wouldn't"}
stop_words = set(stopwords.words('english')) - my_stopwords
ps = PorterStemmer()

# Preprocess the data
corpus = []
for review in df['Review']:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    corpus.append(review)

# Convert text to feature vectors
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = df['Liked']

# Train the model
model = RandomForestClassifier()
model.fit(x, y)

# Streamlit UI
st.title("Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review and find out if it's positive or negative!")

# User input
user_input = st.text_area("Enter your review:")

if st.button("Predict"):
    if user_input:
        # Preprocess the input
        review = re.sub('[^a-zA-Z]', ' ', user_input)
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = ' '.join(review)
        
        # Transform the input using CountVectorizer
        input_data = cv.transform([review]).toarray()
        
        # Predict
        prediction = model.predict(input_data)
        
        # Display result
        if prediction[0] == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review to analyze.")
