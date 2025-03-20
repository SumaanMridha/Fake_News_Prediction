import joblib

# Load the trained model and the fitted vectorizer
model = joblib.load('pipe_.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Load the saved fitted vectorizer

def predict_fake_news(text):
    text_vectorized = vectorizer.transform([text])  # Use the fitted vectorizer
    prediction = model.predict(text_vectorized)
    return 'Fake' if prediction == 1 else 'Real'

if __name__ == "__main__":
    sample_text = "This is an example of fake news."
    print(f"Prediction: {predict_fake_news(sample_text)}")
