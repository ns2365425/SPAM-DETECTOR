import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print(df.head())
def preprocess_text(text):
    text = re.sub(r"\W", " ",text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_message"] = df["message"].apply(preprocess_text)
print(df.head())

vectorizer = TfidfVectorizer(max_features=3000)
x = vectorizer.fit_transform(df["cleaned_message"])
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

def predict_spam(text):
    text = preprocess_text(text)
    text = vectorizer.transform([text])
    prediction = model.predict(text)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# example
text = "Congratulations! You have won a free ticket. Text 'CLAIM' to 123456 to claim your prize."
print(f"Message: {text}")
print(f"Prediction: {predict_spam(text)}")
