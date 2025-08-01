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

def preprocess_text_improved(text):
    """Clean and preprocess text data with better negation handling"""
    if pd.isna(text):
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
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    # Keep negation markers and important words
    tokens = [token for token in tokens if token not in stop_words or token.startswith('NOT_')]
    
    return ' '.join(tokens)

def train_emotion_model_improved():
    """Train the improved emotion classification model"""
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
    
    # Preprocess text with improved negation handling
    print("Preprocessing text with improved negation handling...")
    df['processed_text'] = df['text'].apply(preprocess_text_improved)
    
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
    print("Saving improved model and vectorizer...")
    with open('emotion_model_improved.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('emotion_vectorizer_improved.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Improved emotion model and vectorizer saved successfully!")
    
    # Test the model with some examples
    print("\nTesting improved model with examples:")
    test_texts = [
        "I love this product, it's amazing!",
        "This is terrible, I hate it",
        "The weather is okay today",
        "I'm so happy and excited",
        "This makes me very angry",
        "I'm worried about the future",
        "I feel jealous of their success",
        "app is slow and buggy i dont love it",
        "i dont like this at all",
        "this is not good"
    ]
    
    for text in test_texts:
        processed = preprocess_text_improved(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        print(f"Text: '{text}' -> Emotion: {prediction}")

if __name__ == "__main__":
    train_emotion_model_improved() 