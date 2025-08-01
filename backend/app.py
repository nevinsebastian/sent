from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)

# SQLite DB config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Load the trained emotion model and vectorizer
try:
    with open('emotion_model.pkl', 'rb') as f:
        emotion_model = pickle.load(f)
    with open('emotion_vectorizer.pkl', 'rb') as f:
        emotion_vectorizer = pickle.load(f)
    print("✅ Trained emotion model loaded successfully!")
    print("Available emotions: neutral, love, happiness, sadness, relief, hate, anger, fun, enthusiasm, surprise, empty, worry, boredom")
except FileNotFoundError:
    print("⚠️  Trained emotion model not found. Please run train_emotion_model.py first.")
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

def preprocess_text(text):
    """Clean and preprocess text data with better negation handling"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Handle negations better
    negation_words = ['not', 'dont', 'doesnt', 'didnt', 'wont', 'cant', 'couldnt', 'shouldnt', 'wouldnt', 'havent', 'hasnt', 'hadnt']
    
    # Add negation markers
    for neg_word in negation_words:
        if neg_word in text:
            # Replace negation with a special marker
            text = text.replace(neg_word, 'NOT_')
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s_]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        # Keep negation markers and important words
        tokens = [token for token in tokens if token not in stop_words or token.startswith('NOT_')]
        return ' '.join(tokens)
    except:
        return text

def get_emotion(text):
    """Get emotion using the trained model"""
    if emotion_model is None or emotion_vectorizer is None:
        print("❌ ERROR: Emotion model not loaded!")
        return "neutral"
    
    # Use the trained model
    try:
        processed_text = preprocess_text(text)
        print(f"Original text: {text}")
        print(f"Processed text: {processed_text}")
        
        vectorized_text = emotion_vectorizer.transform([processed_text])
        prediction = emotion_model.predict(vectorized_text)[0]
        
        print(f"Predicted emotion: {prediction}")
        return prediction
    except Exception as e:
        print(f"❌ Error in emotion prediction: {e}")
        return "neutral"

# Routes
@app.route('/')
def home():
    model_status = "✅ Loaded" if emotion_model is not None else "❌ Not found"
    available_emotions = ["neutral", "love", "happiness", "sadness", "relief", "hate", "anger", "fun", "enthusiasm", "surprise", "empty", "worry", "boredom"]
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
