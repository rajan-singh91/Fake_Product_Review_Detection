import pandas as pd
import nltk
import re

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import pickle
import os


# Download stopwords
nltk.download('stopwords')


# Load Dataset
data = pd.read_csv("dataset/reviews.csv")


print("Dataset Loaded")
print(data.head())


# Text Cleaning Function

def clean_text(text):

    text = text.lower()

    text = re.sub(
        '[^a-zA-Z]',
        ' ',
        text
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)



# Apply Cleaning

data["review"] = data["review"].apply(clean_text)


# Input and Output

X = data["review"]

y = data["label"]



# Convert Text into Numbers

vectorizer = TfidfVectorizer()


X = vectorizer.fit_transform(X)



# Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train Model

model = LogisticRegression()


model.fit(
    X_train,
    y_train
)



# Accuracy Check

prediction = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    "Model Accuracy:",
    accuracy
)



# Save Model

if not os.path.exists("model"):
    os.makedirs("model")


pickle.dump(
    model,
    open("model/fake_review_model.pkl","wb")
)


pickle.dump(
    vectorizer,
    open("model/vectorizer.pkl","wb")
)


print("Model Saved Successfully!")