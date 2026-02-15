import zipfile
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 🔹 Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 🔓 Step 1: Unzip archive.zip
with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("data")
print("✅ Unzipped archive.zip into 'data/' folder.")

# 📥 Step 2: Load CSV file from extracted folder
df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = df[['text', 'target']]  # Keep only useful columns

# 🧪 Step 3: Sample only 10,000 tweets for fast training
df = df.sample(10000, random_state=42)

# 🔄 Step 4: Convert target 4 to 1 (Positive), 0 stays as Negative
df['target'] = df['target'].replace(4, 1)

# 🧹 Step 5: Clean the tweet text
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z\s]", "", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

df['clean_text'] = df['text'].apply(clean_text)

# 🧠 Step 6: Prepare data for model
X = df['clean_text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📊 Step 7: Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🧠 Step 8: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 💾 Step 9: Save the model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Training complete! Model and vectorizer saved.")
