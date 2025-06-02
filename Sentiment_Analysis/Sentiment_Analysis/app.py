from flask import Flask, request, render_template
import pickle
from preprocess import preprocess_text

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['review'] # Rahul is a worst person, I want to kill him

    # Preprocess and vectorize
    cleaned_input = preprocess_text(user_input)
    vector_input = vectorizer.transform([cleaned_input])

    # Predict sentiment
    prediction = model.predict(vector_input)[0]
    sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"

    # Return prediction to the frontend
    return render_template('index.html', prediction=sentiment, review=user_input)

if __name__ == '__main__':
    app.run(debug=True)
