# views.py

from django.shortcuts import render
import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import joblib
import os

# Assuming views.py is in the 'yourapp' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the pre-trained model
model_path = os.path.join(BASE_DIR, 'model', 'passmodel.pkl').replace('\\', '/')
vectorizer_path = os.path.join(BASE_DIR, 'model', 'tfidfvectorizer.pkl')

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Create stopwords set
stop = set(stopwords.words('english'))

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 6. Lemmatization
    lemmatize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. Space join words
    return ' '.join(lemmatize_words)

def get_specialist(prediction):
    # Map predictions to specialists
    specialist_mapping = {
        'Acne': 'Skin Specialist Doctor',
        'Birth control': 'Gynecologist',
        'Depression': 'Psychologist',
        'Diabetes type 2': 'Diabetologist',
        'High Blood Pressure': 'Consult Another General Practitioner (GP)'
    }

    return specialist_mapping.get(prediction, 'General Practitioner')

def predict(request):
    raw_text = ""  # Default value
    if request.method == 'POST':
        raw_text = request.POST.get('rawtext', '')

        if raw_text:
            # Text preprocessing
            clean_text = review_to_words(raw_text)

            # Check the number of words in the input
            if len(clean_text.split()) < 3:
                return render(request, 'predict.html', {'rawtext': raw_text, 'result': "We need more information about your problem."})

            # Transform text using the TF-IDF vectorizer
            tfidf_input = vectorizer.transform([clean_text])

            # Check the model's confidence score
            confidence_score = max(model.decision_function(tfidf_input)[0])
            if confidence_score < 0.7:
                return render(request, 'predict.html', {'rawtext': raw_text, 'result': " Wrong input ! Please tell me your disease."})

            # Make a prediction using the model
            prediction = model.predict(tfidf_input)[0]

            # Get specialist suggestion based on prediction
            specialist_suggestion = get_specialist(prediction)

            return render(request, 'predict.html', {'rawtext': raw_text, 'result': prediction, 'specialist_suggestion': specialist_suggestion})
        else:
            raw_text = "There is no text to select"

    return render(request, 'predict.html', {'rawtext': raw_text, 'result': None, 'specialist_suggestion': None})
