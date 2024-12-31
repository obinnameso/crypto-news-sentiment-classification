import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd

# Load the fine-tuned model and tokenizer
model_path = './model_files'  # Replace with your model path
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Map labels to sentiment
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to predict sentiment
def predict_sentiment(text):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    return label_map[predicted_label]

# # Streamlit App
# st.title("Crypto News Sentiment Analysis")
# st.write("Enter a news headline to analyze its sentiment.")

# # Input text box
# user_input = st.text_area("Input Text", placeholder="Type here...")

# if st.button("Analyze Sentiment"):
#     if user_input.strip():
#         sentiment = predict_sentiment(user_input)
#         st.write(f"**Predicted Sentiment:** {sentiment}")
#     else:
#         st.write("Please enter some text to analyze.")


# Function to process a CSV file
def analyze_csv(file):
    """
    Analyzes sentiments for a CSV file containing headlines.

    Args:
        file: Uploaded CSV file.

    Returns:
        DataFrame: Original headlines with predicted sentiments.
    """
    df = pd.read_csv(file)
    if 'headline' not in df.columns:
        st.error("CSV file must contain a 'headline' column.")
        return None
    
    df['Sentiment'] = df['headline'].apply(predict_sentiment)
    return df

# Streamlit App
st.title("Crypto News Sentiment Analysis")
st.write("Enter a text snippet to analyze its sentiment or upload a CSV file with news headlines.")

# Input text box
st.subheader("Analyze Single Headline")
user_input = st.text_area("Input Text", placeholder="Type here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.write("Please enter some text to analyze.")

# CSV file upload
st.subheader("Analyze Multiple headlines in csv File")
uploaded_file = st.file_uploader("Upload a CSV file containing a 'headline' column", type="csv")

if uploaded_file:
    result_df = analyze_csv(uploaded_file)
    if result_df is not None:
        st.write("**Sentiment Analysis Results:**")
        st.dataframe(result_df)

        # Option to download the results
        csv_result = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results",
            data=csv_result,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )