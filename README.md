# Twitter Climate Sentiment Classification

## Project Description

<br>

In this project, we developed a sentiment analysis tool for climate-related tweets, enabling companies to gauge public perception of climate change by their tweets. By analyzing Twitter discussions, businesses can gain insights into how their products or services might be perceived in the context of climate sentiment.

We investigated various supervised machine learning models, including Logistic Regression, Support Vector Machine, Naive Bayes, and Random Forest, to identify the most effective classifier for predicting climate sentiment in tweets. We employed GridSearchCV to optimize the parameters of our chosen model.

<br>

## Feature

This project uses a classification model to determine the sentiment of users' tweets which could either be Pro, Negative, News or Neutral. We also developed a web application using Streamlit for demonstration and easy interaction with our model. 
The web application was designed in a way that it will be able to take csv files and return the sentiments which would suit a use case for an organization seeking to get the sentiment analysis for a group of tweets.

<br>

## Tools used

* Python
* Streamlit
* nltk
* Comet
* GitHub

<br>

## Installation

Step 1: Install Python
Ensure that you have the latest version of Python installed, preferably Python 3.10.11. If you haven't already installed it, you can do so by running the following command:

pip install ipython
Step 2: Download Necessary Corpora and Model
You need to download the required corpora and model to aid with stopword removal and tokenization. Open a Python environment and execute the following commands:

import nltk
nltk.download(['punkt', 'stopwords'])
Step 3: Install Dependencies
Install the project dependencies including pandas, numpy, matplotlib, and scikit-learn using the following command:

pip install -U matplotlib numpy pandas scikit-learn


## Usage

Open your preferred Python environment or notebook.
Import the necessary libraries.
Load the data onto the notebook or import the "clean_train_csv" file directly to skip the cleaning process.
Fit the data into the selected model. The model used for this project is the Support Vector Machine (SVM). You can experiment with different model types and tweak the parameters to suit your requirements.


## Project Structure

The project repository consists of the following folders/files:

train.csv: Contains raw tweets and sentiments used for training the model.
test_with_no_labels.csv: Contains raw tweets without labels, which can be used as a testing dataset.
clean_train.csv: Contains the clean training data. You can load this file directly to skip the cleaning process.
clean_test.csv: Contains the clean test data. You can load this file directly to skip the cleaning process.
