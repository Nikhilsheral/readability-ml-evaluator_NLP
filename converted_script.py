import nltk
nltk.download('stopwords')
nltk.download('punkt')  # For tokenizing sentences and words

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')  # For tokenizing sentences and words

pip install textstat

from nltk.corpus import stopwords
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

pip install nltk

import nltk
nltk.download('stopwords')
nltk.download('punkt')  # For tokenizing sentences and words

# Download stopwords if not already downloaded
nltk.download('stopwords')

import pandas as pd

# Sample dataset with synthetic technical text and feedback scores
data = pd.DataFrame({
    'text': [
        "To install the Linux operating system, start by creating a bootable USB drive. This guide covers each step in detail.",
        "Python functions are defined using the 'def' keyword. The function name is followed by parentheses, which can hold parameters.",
        "In medical terms, hypertension refers to high blood pressure, which can lead to cardiovascular complications if untreated.",
        "This guide provides instructions for setting up a wireless network. Ensure your router is compatible with the latest security protocols.",
        "A class in Python is a blueprint for creating objects, encapsulating data and functions relevant to the object type.",
        "Chronic obstructive pulmonary disease (COPD) is a common lung disease that obstructs airflow from the lungs.",
        "JavaScript enables interactive web pages and is essential for dynamic web applications. Here, we explain fundamental JavaScript concepts.",
        "Diabetes is a chronic disease that affects the way the body processes blood sugar. Early diagnosis is crucial for managing the condition.",
        "The user manual for Model X drone covers assembly, calibration, and troubleshooting of flight-related issues.",
        "SQL is used to manage and manipulate databases. This guide introduces essential SQL queries and operations.",
    ],
    'feedback_score': [3.5, 4.2, 2.8, 3.7, 4.5, 2.9, 4.1, 3.0, 3.8, 4.3]  # Hypothetical feedback on readability for older adults
})


# Import necessary libraries
from nltk.corpus import stopwords
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Preprocess text data (you may include tokenization, removing stopwords, etc., as needed)
data['processed_text'] = data['text'].apply(lambda x: x.lower())  # Simple lowercase as an example

# Extract readability metrics as features
data['flesch_kincaid'] = data['text'].apply(flesch_kincaid_grade)
data['smog'] = data['text'].apply(smog_index)
data['gunning_fog'] = data['text'].apply(gunning_fog)
data['dale_chall'] = data['text'].apply(dale_chall_readability_score)

# Define features (readability scores) and target (feedback score)
X = data[['flesch_kincaid', 'smog', 'gunning_fog', 'dale_chall']]
y = data['feedback_score']

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model (or another model of your choice)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


pip install xgboost

import os
print(os.getcwd())


# Import necessary libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load your dataset
data = pd.read_csv('your_dataset.csv')  # Update the path to your dataset

# Check if the 'text' column exists
if 'text' not in data.columns:
    raise ValueError("The dataset must contain a 'text' column.")

# Preprocess text data
def preprocess_text(text):
    text = text.lower()  # Lowercase
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

data['processed_text'] = data['text'].apply(preprocess_text)

# Extract readability metrics as features
data['flesch_kincaid'] = data['text'].apply(flesch_kincaid_grade)
data['smog'] = data['text'].apply(smog_index)
data['gunning_fog'] = data['text'].apply(gunning_fog)
data['dale_chall'] = data['text'].apply(dale_chall_readability_score)

# Define features (readability scores) and target (feedback score)
X = data[['flesch_kincaid', 'smog', 'gunning_fog', 'dale_chall']]
y = data['feedback_score']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Use XGBoost Regressor
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared as an additional metric
r_squared = best_model.score(X_test, y_test)
print(f"R-squared: {r_squared}")

# Import necessary libraries
import pandas as pd
from nltk.corpus import stopwords
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv('your_dataset.csv')  # Update the path to your dataset

# Check if the 'text' column exists
if 'text' not in data.columns:
    raise ValueError("The dataset must contain a 'text' column.")

# Preprocess text data (improve preprocessing as needed)
def preprocess_text(text):
    text = text.lower()  # Lowercase
    # Add more preprocessing steps (e.g., tokenization, removing stopwords)
    return text

data['processed_text'] = data['text'].apply(preprocess_text)

# Extract readability metrics as features
data['flesch_kincaid'] = data['text'].apply(flesch_kincaid_grade)
data['smog'] = data['text'].apply(smog_index)
data['gunning_fog'] = data['text'].apply(gunning_fog)
data['dale_chall'] = data['text'].apply(dale_chall_readability_score)

# Define features (readability scores) and target (feedback score)
X = data[['flesch_kincaid', 'smog', 'gunning_fog', 'dale_chall']]
y = data['feedback_score']

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use XGBoost Regressor
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared as an additional metric
r_squared = best_model.score(X_test, y_test)
print(f"R-squared: {r_squared}")


#streamlit

import nltk
import pandas as pd
import streamlit as st
import random
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    # Add more preprocessing as needed
    return text

# Function to classify categories - placeholder
def classify_category(text):
    categories = ['Technology', 'Health', 'Science', 'Finance']
    return random.choice(categories)

# Function to calculate readability scores
def calculate_readability(text):
    return {
        'Flesch-Kincaid Grade Level': flesch_kincaid_grade(text),
        'SMOG Index': smog_index(text),
        'Gunning Fog Index': gunning_fog(text),
        'Dale-Chall Score': dale_chall_readability_score(text)
    }

# Improved function to simplify text for age groups
def simplify_text(text, age_group):
    if age_group < 13:
        simplified = f"Simplified for children: {text[:100]}..."
    elif 13 <= age_group < 18:
        simplified = f"Simplified for teenagers: {text[:200]}..."
    else:
        simplified = f"Simplified for adults: {text[:300]}..."
    return simplified, text  # returning full text for "see more" functionality

# Load synthetic dataset and additional features for training
data = pd.DataFrame({
    'text': [
        "To install the Linux operating system, start by creating a bootable USB drive.",
        "Python functions are defined using the 'def' keyword.",
        "In medical terms, hypertension refers to high blood pressure.",
        "This guide provides instructions for setting up a wireless network.",
        "A class in Python is a blueprint for creating objects.",
        "Chronic obstructive pulmonary disease (COPD) is a common lung disease.",
        "JavaScript enables interactive web pages.",
        "Diabetes is a chronic disease that affects blood sugar processing.",
        "The user manual for Model X drone covers assembly.",
        "SQL is used to manage and manipulate databases."
    ],
    'feedback_score': [3.5, 4.2, 2.8, 3.7, 4.5, 2.9, 4.1, 3.0, 3.8, 4.3]
})

# Feature extraction
data['processed_text'] = data['text'].apply(preprocess_text)
data['flesch_kincaid'] = data['text'].apply(flesch_kincaid_grade)
data['smog'] = data['text'].apply(smog_index)
data['gunning_fog'] = data['text'].apply(gunning_fog)
data['dale_chall'] = data['text'].apply(dale_chall_readability_score)

X = data[['flesch_kincaid', 'smog', 'gunning_fog', 'dale_chall']]
y = data['feedback_score']

# Train-test split and model training with improved hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {model.score(X_test, y_test)}")




pip install streamlit





import matplotlib.pyplot as plt
import seaborn as sns



import streamlit as st
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
import pandas as pd
import random  # For category simulation

# Function to classify category based on file content (Placeholder function)
def classify_category(text):
    # Placeholder: Replace this with your trained model's category classifier
    categories = ['Technology', 'Health', 'Science', 'Finance']
    return random.choice(categories)

# Function to calculate readability scores
def calculate_readability(text):
    return {
        'Flesch-Kincaid Grade Level': flesch_kincaid_grade(text),
        'SMOG Index': smog_index(text),
        'Gunning Fog Index': gunning_fog(text),
        'Dale-Chall Score': dale_chall_readability_score(text)
    }

# Function to simplify text based on age group (Placeholder for NLP simplification logic)
def simplify_text(text, age_group):
    # Add logic here to use NLP models for text simplification
    if age_group < 13:
        return f"Simplified for children: {text[:200]}... (more simplified text here)"
    elif 13 <= age_group < 18:
        return f"Simplified for teenagers: {text[:400]}... (more simplified text here)"
    else:
        return f"Simplified for adults: {text[:600]}... (more simplified text here)"

st.title("Multi-File Readability and Simplification for Different Age Groups")

# Step 1: Upload multiple files
uploaded_files = st.file_uploader("Upload multiple text files", accept_multiple_files=True, type=["txt", "md"])

# Check if files are uploaded
if uploaded_files:
    file_data = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        category = classify_category(text)
        readability_scores = calculate_readability(text)
        file_data.append({
            'file_name': uploaded_file.name,
            'text': text,
            'category': category,
            'readability_scores': readability_scores
        })
    
    # Step 2: Display file categories and readability difficulty for different age groups
    st.write("### File Categories and Readability Levels:")
    for file in file_data:
        st.write(f"**File**: {file['file_name']}")
        st.write(f"**Category**: {file['category']}")
        st.write("**Readability Scores:**")
        for metric, score in file['readability_scores'].items():
            st.write(f"- {metric}: {score}")
        st.write("---")
    
    # Step 3: User input for age group and file selection
    age_group = st.number_input("Which age group do you belong to?", min_value=5, max_value=100, step=1)
    file_selection = st.selectbox("Select the file you want to understand:", [f['file_name'] for f in file_data])
    
    # Step 4: Simplify selected file for the given age group
    if st.button("Generate Simplified Content"):
        selected_file = next(f for f in file_data if f['file_name'] == file_selection)
        simplified_text = simplify_text(selected_file['text'], age_group)
        st.write("### Simplified Content")
        st.write(simplified_text)


#command: streamlit run readability_app.py

import streamlit as st

st.title("Multi-File Readability and Simplification for Different Age Groups")
st.title("Hello, Streamlit!"


# Streamlit interface for file upload and content simplification
import streamlit as st
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
import pandas as pd
import random  # For category simulation

st.title("Multi-File Readability and Simplification for Different Age Groups")
uploaded_files = st.file_uploader("Upload multiple text files", accept_multiple_files=True, type=["txt", "md"])

if uploaded_files:
    file_data = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        category = classify_category(text)
        readability_scores = calculate_readability(text)
        file_data.append({
            'file_name': uploaded_file.name,
            'text': text,
            'category': category,
            'readability_scores': readability_scores
        })
    
    st.write("### File Categories and Readability Levels:")
    for file in file_data:
        st.write(f"**File**: {file['file_name']}")
        st.write(f"**Category**: {file['category']}")
        st.write("**Readability Scores:**")
        for metric, score in file['readability_scores'].items():
            st.write(f"- {metric}: {score}")
        st.write("---")
    
    age_group = st.number_input("Which age group do you belong to?", min_value=5, max_value=100, step=1)
    file_selection = st.selectbox("Select the file you want to understand:", [f['file_name'] for f in file_data])
    
    if st.button("Generate Simplified Content"):
        selected_file = next(f for f in file_data if f['file_name'] == file_selection)
        simplified_text, full_text = simplify_text(selected_file['text'], age_group)
        st.write("### Simplified Content")
        st.write(simplified_text)
        
        # "See more" functionality
        if st.button("See more"):
            st.write(full_text)

#streamlit run readability_app.py

