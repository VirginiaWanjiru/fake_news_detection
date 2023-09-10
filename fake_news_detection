import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('fake_news.csv')

# Clean the data
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace('[^a-zA-Z0-9]', ' ')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Create a logistic regression model
model = LogisticRegression()
model.fit(X, df['label'])

# Evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2)
score = accuracy_score(model.predict(X_test), y_test)
print('Accuracy:', score)

# Make predictions
predictions = model.predict(X)

# Save the predictions to a file
df['prediction'] = predictions
df.to_csv('fake_news_predictions.csv')