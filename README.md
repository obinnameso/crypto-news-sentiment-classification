# Crypto News Sentiment Classification

## Project Overview

This project focuses on developing a sentiment analysis tool for crypto-related news headlines, providing insights into public perception. Users can determine whether a given headline expresses a Positive, Negative, or Neutral sentiment. A user-friendly web application, built using Streamlit, demonstrates the tool’s capabilities and allows for easy interaction with the fine-tuned model.

<br>

## Key Features

* Sentiment Analysis: Utilizes a Large Language Model (LLM) to classify the sentiment of news headlines.
* Versatile Input Handling: Supports single news headlines and batch processing via CSV files, catering to both individual users and organizational use cases.
* Streamlit Web Application: Interactive web interface for demonstrating the model, designed for ease of use and accessibility.
* Real-world Data: Trained and tested on datasets sourced from Kaggle <a href = 'https://www.kaggle.com/datasets/kaballa/cryptoner-ml-model?select=articlesData.csv'>(link to dataset)<a>. <br>

## Tools used

* Python
* Streamlit
* Transformers (for LLM implementation)
* Torch
* nltk (Natural Language Toolkit for preprocessing)
* GitHub (version control)


## Installation

1. pip install python
2. Download Necessary Corpora and Model <br> 
You need to download the required corpora and model to aid with stopword removal and tokenization. Open a Python environment and execute the following commands: <br> 
import nltk <br> 
nltk.download(['punkt', 'stopwords','wordnet','punkt_tab'])

3. Install the project dependencies including pandas, numpy, matplotlib, transformers, torch and scikit-learn using the following command: <br> 
pip install matplotlib numpy pandas scikit-learn transformers torch accelerate datasets streamlit -U
 <br> 

## How to Use

1. Launch the Streamlit web application.
2. Input either a single news headline or upload a CSV file containing multiple headlines.
3. View the sentiment analysis results (Positive, Negative, or Neutral).

This tool is designed to assist users in understanding the sentiment trends in crypto-related news, supporting better decision-making.
 



