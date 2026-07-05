import pandas as pd
import pickle
import matplotlib.pyplot as plt

from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("dataset/train.csv")

# Merge Title + Description
df["text"] = df["Title"] + " " + df["Description"]

# Clean Text
df["text"] = df["text"].apply(clean_text)

# Features & Labels
X = df["text"]
y = df["Class Index"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save Model
pickle.dump(
    model,
    open("model/news_model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("model/vectorizer.pkl", "wb")
)

print("Model Saved Successfully!")

# Accuracy Graph
plt.figure(figsize=(6,4))

plt.bar(
    ["Accuracy"],
    [accuracy]
)

plt.title("News Classification Accuracy")
plt.ylabel("Score")
plt.ylim(0,1)

plt.savefig("accuracy.png")

print("Accuracy Graph Saved!")