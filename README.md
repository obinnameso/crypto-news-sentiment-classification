# Crypto News Sentiment Classification

## Project Description

In this project, we developed a sentiment analysis tool for crypto-related news headlines, enabling users to gauge public perception of crypto news headlines.

<br>

## Feature

This project uses an LLM to determine the sentiment of news headlines which could either be Positive, Negative or Neutral. I also developed a web application using Streamlit for demonstration and easy interaction with the fine-tuned model. 
The web application was designed in a way that it will be able to take both single headlines or csv files also and return the sentiments which would suit a use case for an organization seeking to get the sentiment analysis for a group of newws headlines.
<br>

## Tools used

* Python
* Streamlit
* Transformers
* Torch
* nltk
* GitHub


## Installation

1. pip install python
2. Download Necessary Corpora and Model <br> 
You need to download the required corpora and model to aid with stopword removal and tokenization. Open a Python environment and execute the following commands: <br> 
import nltk <br> 
nltk.download(['punkt', 'stopwords'])

3. Install the project dependencies including pandas, numpy, matplotlib, transformers, torch and scikit-learn using the following command: <br> 
pip install matplotlib numpy pandas scikit-learn transformers torch accelerate datasets streamlit -U
 <br> 

## Result 
 



