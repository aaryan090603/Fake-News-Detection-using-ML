# Fake-News-Detection-using-ML
Table of Contents
Importing the Dependencies
About the Dataset
Downloading Stopwords
Printing Stopwords
Loading the Dataset
Inspecting the Data
Handling Missing Values
Merging Author and Title
Separating Features and Labels
Stemming
Applying Stemming
Preparing Data for Training
Text Vectorization
Splitting the Data
Model Training
Model Evaluation
1. Importing the Dependencies
In this section, we import all the necessary libraries for data manipulation, natural language processing (NLP), and machine learning.

python
Copy code
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
2. About the Dataset
The dataset used in this project contains the following columns:

id: Unique identifier for each news article.
title: The title of the news article.
author: The author of the news article.
text: The main content of the article, which could be incomplete.
label: Indicates whether the news article is real (0) or fake (1).
3. Downloading Stopwords
Stopwords are commonly used words that are usually removed during text processing.

python
Copy code
import nltk
nltk.download('stopwords')
4. Printing Stopwords
This code prints the list of English stopwords.

python
Copy code
print(stopwords.words('english'))
5. Loading the Dataset
We load the dataset into a Pandas DataFrame.

python
Copy code
news_dataset = pd.read_csv('/content/train.csv')
6. Inspecting the Data
We inspect the shape of the dataset and print the first few rows to understand its structure.

python
Copy code
news_dataset.shape
python
Copy code
news_dataset.head()
7. Handling Missing Values
We count the number of missing values in each column and then replace them with empty strings.

python
Copy code
news_dataset.isnull().sum()
python
Copy code
news_dataset = news_dataset.fillna('')
8. Merging Author and Title
We combine the author and title columns to create a new content column.

python
Copy code
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
9. Separating Features and Labels
We separate the features (input data) and labels (output data).

python
Copy code
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
10. Stemming
Stemming reduces words to their root form. For example, "running" and "runner" both reduce to "run".

python
Copy code
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
11. Applying Stemming
We apply the stemming function to the content column.

python
Copy code
news_dataset['content'] = news_dataset['content'].apply(stemming)
12. Preparing Data for Training
We separate the processed content and label columns into feature and target variables.

python
Copy code
X = news_dataset['content'].values
Y = news_dataset['label'].values
13. Text Vectorization
We convert the text data into numerical data using TF-IDF vectorization.

python
Copy code
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
14. Splitting the Data
We split the data into training and testing sets.

python
Copy code
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
15. Model Training
We train a logistic regression model on the training data.

python
Copy code
model = LogisticRegression()
model.fit(X_train, Y_train)
16. Model Evaluation
We evaluate the model's performance on the training and testing sets using accuracy.

python
Copy code
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Training Accuracy: {train_accuracy}")

test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Testing Accuracy: {test_accuracy}")

This concludes the detailed explanation of the code with a table of contents and descriptions for each section. 
