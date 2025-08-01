from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

# SQLite DB config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Load the trained emotion model and vectorizer
try:
    with open('emotion_model_final.pkl', 'rb') as f:
        emotion_model = pickle.load(f)
    with open('emotion_vectorizer_final.pkl', 'rb') as f:
        emotion_vectorizer = pickle.load(f)
    print("✅ Improved emotion model loaded successfully!")
    print("Available emotions: anger, boredom, empty, enthusiasm, fun, happiness, hate, love, neutral, relief, sadness, surprise, worry")
except FileNotFoundError:
    print("⚠️  Improved emotion model not found. Please run train_emotion_model_final.py first.")
    emotion_model = None
    emotion_vectorizer = None

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(120))
    name = db.Column(db.String(120))

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    text = db.Column(db.Text)
    emotion = db.Column(db.String(20))  # Changed from sentiment to emotion

def preprocess_text_advanced(text):
    """Advanced text preprocessing with better negation and sentiment handling"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Handle contractions
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "can't": "cannot", "couldn't": "could not",
        "shouldn't": "should not", "wouldn't": "would not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Handle negations with better context - mark the entire phrase as negated
    negation_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'hardly', 'barely', 'scarcely']
    
    # Split into words and mark negated phrases
    words = text.split()
    processed_words = []
    negate_next = False
    
    for i, word in enumerate(words):
        if word in negation_words:
            negate_next = True
            processed_words.append('NOT_')
        elif negate_next:
            # Mark the next few words as negated
            processed_words.append(f'NOT_{word}')
            negate_next = False
        else:
            processed_words.append(word)
    
    text = ' '.join(processed_words)
    
    # Remove special characters but keep important punctuation for sentiment
    text = re.sub(r'[^a-zA-Z\s_!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        # Keep negation markers and important sentiment words
        tokens = [token for token in tokens if token not in stop_words or token.startswith('NOT_') or len(token) > 2]
        return ' '.join(tokens)
    except:
        return text

def extract_sentiment_features(text):
    """Extract additional sentiment features using VADER"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return {
        'vader_compound': scores['compound'],
        'vader_positive': scores['pos'],
        'vader_negative': scores['neg'],
        'vader_neutral': scores['neu']
    }

def get_emotion(text):
    """Get emotion using the improved trained model with negation correction"""
    if emotion_model is None or emotion_vectorizer is None:
        print("❌ ERROR: Emotion model not loaded!")
        return "neutral"
    
    # Use the improved trained model
    try:
        processed_text = preprocess_text_advanced(text)
        print(f"Original text: {text}")
        print(f"Processed text: {processed_text}")
        
        # Vectorize text
        vectorized_text = emotion_vectorizer.transform([processed_text])
        
        # Extract sentiment features
        sentiment_features = extract_sentiment_features(text)
        sentiment_array = np.array([[sentiment_features['vader_compound'], 
                                   sentiment_features['vader_positive'], 
                                   sentiment_features['vader_negative'], 
                                   sentiment_features['vader_neutral']]])
        
        # Combine text features with sentiment features
        combined_features = hstack([vectorized_text, sentiment_array])
        
        prediction = emotion_model.predict(combined_features)[0]
        
        # Post-processing: Check for negation conflicts
        vader_compound = sentiment_features['vader_compound']
        
        # If VADER says negative but model predicts positive emotion, adjust
        if vader_compound < -0.3:  # Strongly negative
            if prediction in ['love', 'happiness', 'fun', 'enthusiasm']:
                # Convert to appropriate negative emotion
                if 'hate' in text.lower() or 'angry' in text.lower():
                    prediction = 'hate'
                elif 'sad' in text.lower():
                    prediction = 'sadness'
                elif 'worr' in text.lower():
                    prediction = 'worry'
                else:
                    prediction = 'anger'
        
        # If VADER says positive but model predicts negative emotion, adjust
        elif vader_compound > 0.3:  # Strongly positive
            if prediction in ['hate', 'anger', 'sadness', 'worry']:
                # Convert to appropriate positive emotion
                if 'love' in text.lower():
                    prediction = 'love'
                elif 'happy' in text.lower():
                    prediction = 'happiness'
                elif 'fun' in text.lower():
                    prediction = 'fun'
                else:
                    prediction = 'happiness'
        
        print(f"VADER compound: {vader_compound:.3f}")
        print(f"Predicted emotion: {prediction}")
        return prediction
    except Exception as e:
        print(f"❌ Error in emotion prediction: {e}")
        return "neutral"

# Routes
@app.route('/')
def home():
    model_status = "✅ Loaded" if emotion_model is not None else "❌ Not found"
    available_emotions = ["anger", "boredom", "empty", "enthusiasm", "fun", "happiness", "hate", "love", "neutral", "relief", "sadness", "surprise", "worry"]
    return jsonify({
        "message": "Flask API is running!", 
        "emotion_model": model_status,
        "available_emotions": available_emotions,
        "endpoints": ["/signup", "/login", "/feedback", "/admin/feedbacks", "/emotion"]
    })

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    new_user = User(email=data['email'], password=data['password'], name=data['name'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully!"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email'], password=data['password']).first()
    if user:
        return jsonify({"message": "Login successful", "user_id": user.id})
    else:
        return jsonify({"message": "Invalid credentials"}), 401

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    text = data['text']
    emotion = get_emotion(text)
    feedback = Feedback(user_id=data['user_id'], text=text, emotion=emotion)
    db.session.add(feedback)
    db.session.commit()
    return jsonify({"message": "Feedback submitted", "emotion": emotion})

@app.route('/admin/feedbacks', methods=['GET'])
def get_all_feedback():
    feedbacks = Feedback.query.all()
    result = [{"id": f.id, "user_id": f.user_id, "text": f.text, "emotion": f.emotion} for f in feedbacks]
    return jsonify(result)

@app.route('/emotion', methods=['POST'])
def analyze_emotion():
    """New endpoint to analyze emotion of any text"""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Text field is required"}), 400
    
    text = data['text']
    emotion = get_emotion(text)
    
    return jsonify({
        "text": text,
        "emotion": emotion,
        "model_used": "trained_model" if emotion_model is not None else "fallback"
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
