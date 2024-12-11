from flask import Flask, request, render_template, session, redirect, url_for
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
import os

# Download necessary NLTK data (Only required the first time)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Necessary for session to work

# Load pre-trained model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Initialize the PorterStemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    
    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    text = [i for i in text if i not in stop_words and i not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    # Check if there is a previous result in the session, clear it on GET request
    result = session.get('result', None)
    if request.method == "POST":
        input_sms = request.form.get("message")
        
        if input_sms and input_sms.strip():  # Ensure input is not empty or just whitespace
            # Preprocess text
            transformed_sms = transform_text(input_sms)
            
            # Vectorize text
            vector_input = tfidf.transform([transformed_sms])
            
            # Predict result
            prediction = model.predict(vector_input)[0]
            session['result'] = prediction  # Store result in session
        else:
            session['result'] = "empty"  # Store 'empty' if no input text is given
        
        # Redirect to avoid showing result after refreshing
        return redirect(url_for('index'))
    
    # Clear result after rendering page (on GET request)
    session.pop('result', None)
    
    return render_template("index.html", result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run()
