import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
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

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def preprocess_text_advanced(text):
    """Advanced text preprocessing with better negation and sentiment handling"""
    if pd.isna(text):
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
    
    # Handle negations with better context
    negation_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'hardly', 'barely', 'scarcely']
    
    # Add negation markers
    for neg_word in negation_words:
        if neg_word in text:
            text = text.replace(neg_word, 'NOT_')
    
    # Remove special characters but keep important punctuation for sentiment
    text = re.sub(r'[^a-zA-Z\s_!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    # Keep negation markers and important sentiment words
    tokens = [token for token in tokens if token not in stop_words or token.startswith('NOT_') or len(token) > 2]
    
    return ' '.join(tokens)

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

def train_emotion_classification_model():
    """Train an emotion classification model for all 13 emotions"""
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
    print("Preprocessing text with advanced techniques...")
    df['processed_text'] = df['text'].apply(preprocess_text_advanced)
    
    # Check emotion distribution
    emotion_counts = df['Emotion'].value_counts()
    print(f"Emotion distribution:\n{emotion_counts}")
    
    # Show all unique emotions
    unique_emotions = df['Emotion'].unique()
    print(f"\nAll unique emotions ({len(unique_emotions)}): {sorted(unique_emotions)}")
    
    # Extract additional sentiment features
    print("Extracting sentiment features...")
    sentiment_features = df['text'].apply(extract_sentiment_features)
    df['vader_compound'] = sentiment_features.apply(lambda x: x['vader_compound'])
    df['vader_positive'] = sentiment_features.apply(lambda x: x['vader_positive'])
    df['vader_negative'] = sentiment_features.apply(lambda x: x['vader_negative'])
    df['vader_neutral'] = sentiment_features.apply(lambda x: x['vader_neutral'])
    
    # Prepare features and target
    X_text = df['processed_text']
    y = df['Emotion']
    
    # Split the data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train_text)}")
    print(f"Test set size: {len(X_test_text)}")
    
    # Vectorize text with better parameters for emotion classification
    print("Vectorizing text with improved parameters...")
    vectorizer = TfidfVectorizer(
        max_features=15000, 
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vectorized = vectorizer.fit_transform(X_train_text)
    X_test_vectorized = vectorizer.transform(X_test_text)
    
    # Get sentiment features for train/test
    train_sentiment_features = df.loc[X_train_text.index, ['vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']].values
    test_sentiment_features = df.loc[X_test_text.index, ['vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral']].values
    
    # Combine text features with sentiment features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_vectorized, train_sentiment_features])
    X_test_combined = hstack([X_test_vectorized, test_sentiment_features])
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Train multiple models and compare
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42, 
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            class_weight=class_weight_dict,
            random_state=42,
            max_iter=1000,
            C=1.0
        )
    }
    
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='f1_macro')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set
        model.fit(X_train_combined, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_combined)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Check if this is the best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with F1 Macro: {best_f1:.4f}")
    
    # Save the best model and vectorizer
    print("Saving best model and vectorizer...")
    with open('emotion_model_final.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('emotion_vectorizer_final.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("Final emotion model and vectorizer saved successfully!")
    
    # Test the model with examples
    print("\nTesting final emotion model with examples:")
    test_texts = [
        "I love this product, it's amazing!",
        "This is terrible, I hate it",
        "The weather is okay today",
        "I'm so happy and excited",
        "This makes me very angry",
        "I'm worried about the future",
        "I feel jealous of their success",
        "I'm scared of the dark",
        "This disgusts me",
        "I feel surprised by the news",
        "I'm relieved that it's over",
        "I feel empty inside",
        "This is fun and enjoyable"
    ]
    
    for text in test_texts:
        processed = preprocess_text_advanced(text)
        vectorized = vectorizer.transform([processed])
        sentiment_features = extract_sentiment_features(text)
        sentiment_array = np.array([[sentiment_features['vader_compound'], 
                                   sentiment_features['vader_positive'], 
                                   sentiment_features['vader_negative'], 
                                   sentiment_features['vader_neutral']]])
        combined = hstack([vectorized, sentiment_array])
        prediction = best_model.predict(combined)[0]
        print(f"Text: '{text}' -> Emotion: {prediction}")

if __name__ == "__main__":
    train_emotion_classification_model() 