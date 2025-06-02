import pandas as pd   # read the file
import pickle   # after training a model, if we want to use that model in webapp, 
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import accuracy_score
from preprocess import preprocess_text

# Load dataset
df = pd.read_csv('Test.csv')  
df = df[['text', 'label']].dropna()  # remove the missing data rows

# Preprocess text
df['cleaned_review'] = df['text'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
