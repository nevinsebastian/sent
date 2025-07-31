from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

# SQLite DB config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

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
    sentiment = db.Column(db.String(20))

# Routes
@app.route('/')
def home():
    return jsonify({"message": "Flask API is running!", "endpoints": ["/signup", "/login", "/feedback", "/admin/feedbacks"]})

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
    sentiment = get_sentiment(text)
    feedback = Feedback(user_id=data['user_id'], text=text, sentiment=sentiment)
    db.session.add(feedback)
    db.session.commit()
    return jsonify({"message": "Feedback submitted", "sentiment": sentiment})

@app.route('/admin/feedbacks', methods=['GET'])
def get_all_feedback():
    feedbacks = Feedback.query.all()
    result = [{"id": f.id, "user_id": f.user_id, "text": f.text, "sentiment": f.sentiment} for f in feedbacks]
    return jsonify(result)

# Sentiment function
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
