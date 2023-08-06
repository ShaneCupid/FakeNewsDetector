#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load data
@st.cache
def load_data():
    data = pd.read_csv('fake.csv')
    data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    data = data.drop("label", axis=1)
    return data['text'], data['fake']

X, y = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorize data
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)

# Streamlit user interface
st.title('Fake News Classifier')

user_input = st.text_area("Enter a news article:")

if st.button('Predict'):
    vectorized_input = vectorizer.transform([user_input])
    prediction = clf.predict(vectorized_input)
    if prediction == 0:
        st.write("This appears to be a real news article.")
    else:
        st.write("This appears to be a fake news article.")
