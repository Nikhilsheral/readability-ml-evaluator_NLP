import streamlit as st
from textstat import flesch_kincaid_grade, smog_index, gunning_fog, dale_chall_readability_score
import pandas as pd

# Function to classify category based on keywords
def classify_category(text):
    text = text.lower()
    if "sports" in text or "athlete" in text or "competition" in text:
        return "Sports"
    elif "health" in text or "wellness" in text or "exercise" in text:
        return "Health"
    elif "technology" in text or "innovation" in text:
        return "Technology"
    elif "finance" in text or "money" in text:
        return "Finance"
    elif "politics" in text or "money" in text:
        return "Politics"
    elif "fitness" in text or "money" in text:
        return "fitness"
    else:
        return "General"

# Function to calculate readability scores
def calculate_readability(text):
    return {
        'Flesch-Kincaid Grade Level': flesch_kincaid_grade(text),
        'SMOG Index': smog_index(text),
        'Gunning Fog Index': gunning_fog(text),
        'Dale-Chall Score': dale_chall_readability_score(text)
    }

# Function to simplify text based on age group
def simplify_text(text, age_group):
    if age_group < 13:
        return text.replace("essential", "very important").replace("pursuits", "activities")[:600]  # Example replacements
    elif 13 <= age_group < 18:
        return text.replace("essential", "important").replace("pursuits", "activities")[:1200]  # More text for teenagers
    else:
        return text.replace("essential", "important").replace("pursuits", "activities")[:2000]  # Further expand for adults

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


#streamlit run readability_app.py