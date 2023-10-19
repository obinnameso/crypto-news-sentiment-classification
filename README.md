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
* Sklearn
* Comet
* GitHub
<br>

## Installation

1. pip install ipython (preferably Python 3.10.11)
2. Download Necessary Corpora and Model <br> 
You need to download the required corpora and model to aid with stopword removal and tokenization. Open a Python environment and execute the following commands: <br> 
import nltk <br> 
nltk.download(['punkt', 'stopwords'])

3. Install the project dependencies including pandas, numpy, matplotlib, and scikit-learn using the following command: <br> 
pip install -U matplotlib numpy pandas scikit-learn

## Result 
Having done a thorough EDA on the dataset, we visualized the amount of data we had on the given sentiments as shown in the chart below: <br> 

<p align="center">
<img align = "center" width="800" height="500" src="https://github.com/obinnameso/twitter-climate-sentiment-classification/blob/main/images/sentiment_dist_pie_chart.png?raw=true">
</p> <br> 

We also found the highest occuring words in the dataset as shown in the graph below: <br> 

<p align="center">
<img align = "center" width="800" height="600" src="https://github.com/obinnameso/twitter-climate-sentiment-classification/blob/main/images/copy_highest_occuring_words.png?raw=true">
</p> 
<br> 

Using GridSearchCV and Sklearn's pipeline class, we performed hyper parameter tunning on the Support Vector and Random forest classifiers. This allowed us to identify the best parameters for the vectorizer and the classifer as well. With an accuracy score of 0.718, our model performed fairly.

## Recommendation

This project has shed light on public perceptions of climate change, providing valuable insights for businesses. By harnessing machine learning techniques, we were able to extract actionable insights that can guide companies' market research and strategic decisions. Here are key takeaways from this endeavor:

* Market Research Insights: This project offers valuable market research insights for businesses. By accurately classifying individuals' beliefs on climate change, companies gain access to a broad spectrum of consumer sentiment. This data can inform their marketing strategies, helping them develop products and services that align with customers' environmental concerns and increase their market share.

* Competitive Advantage: Our project provides a competitive edge for companies in the growing market of environmentally friendly and sustainable products. By leveraging machine learning techniques, we help businesses stay ahead of their competitors by understanding consumer perceptions and preferences in real-time. This enables them to tailor their offerings, attracting a growing segment of environmentally conscious consumers.

* Public Awareness and Concern: Our analysis reveals a significant portion of individuals expressing strong beliefs in climate change and its potential impacts. This heightened awareness underscores the urgency for businesses to incorporate sustainability and environmental consciousness into their offerings.

## Acknowledgement
I would like to also ackowledge my dynamic team of data and innovation enthusiasts, who worked together to drive positive impact through advanced data analysis and machine learning while working on the project:

* Karabo Lamola
* Mukhtar Abebefe
* Sandile Mdluli
* Chidinma Madukife
* Greensmill Akpa
* Obot Joshua
<br> 

### Note:
* Due to the nature of this project, code cannot be shared publicly.
* Twitter as at the time of writing is known as X.

