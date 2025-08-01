import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def train_emotion_model():
    """Train the emotion classification model"""
    print("Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv('emotion_sentiment_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    print(f"Missing values in text: {df['text'].isnull().sum()}")
    print(f"Missing values in Emotion: {df['Emotion'].isnull().sum()}")
    
    # Remove rows with missing values
    df = df.dropna(subset=['text', 'Emotion'])
    
    print(f"Dataset shape after removing missing values: {df.shape}")
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Check emotion distribution
    emotion_counts = df['Emotion'].value_counts()
    print(f"Emotion distribution:\n{emotion_counts}")
    
    # Prepare features and target
    X = df['processed_text']
    y = df['Emotion']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_vectorized, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vectorized)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    with open('emotion_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('emotion_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Emotion model and vectorizer saved successfully!")
    
    # Test the model with some examples
    print("\nTesting model with examples:")
    test_texts = [
        "I love this product, it's amazing!",
        "This is terrible, I hate it",
        "The weather is okay today",
        "I'm so happy and excited",
        "This makes me very angry",
        "I'm worried about the future",
        "I feel jealous of their success"
    ]
    
    for text in test_texts:
        processed = preprocess_text(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        print(f"Text: '{text}' -> Emotion: {prediction}")

if __name__ == "__main__":
    train_emotion_model() 