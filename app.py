import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

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

# Streamlit App
st.title("Crypto News Sentiment Analysis")
st.write("Enter a news headline to analyze its sentiment.")

# Input text box
user_input = st.text_area("Input Text", placeholder="Type here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.write("Please enter some text to analyze.")
