import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')  # For tokenizing sentences and words

from nltk.corpus import stopwords
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Sample data - Replace this with the actual dataset
data = pd.DataFrame({
    'text': [
        "Technical manual example 1...",
        "User guide example 2...",
        # Add more entries
    ],
    'feedback_score': [3.2, 4.5]  # User feedback score for readability
})

# Preprocess text by removing stopwords and calculating readability scores
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Apply preprocessing
data['processed_text'] = data['text'].apply(preprocess_text)

# Calculate readability features
def calculate_features(text):
    return {
        'flesch_kincaid': flesch_kincaid_grade(text),
        'smog_index': smog_index(text),
        'gunning_fog': gunning_fog(text),
        'dale_chall': dale_chall_readability_score(text),
        'sentence_length': len(nltk.sent_tokenize(text)),
        'word_count': len(nltk.word_tokenize(text))
    }

# Extract features
features = data['text'].apply(calculate_features)
features_df = pd.DataFrame(features.tolist())
data = pd.concat([data, features_df], axis=1)

# Define features (X) and target (y)
X = data[['flesch_kincaid', 'smog_index', 'gunning_fog', 'dale_chall', 'sentence_length', 'word_count']]
y = data['feedback_score']
