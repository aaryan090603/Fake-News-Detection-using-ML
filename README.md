# Fake-News-Detection-using-ML
Here's a comprehensive explanation of the README file , outlining each part of the fake news detection project using the dataset.

---

# Fake News Detection

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Prediction](#prediction)
8. [Conclusion](#conclusion)

## Introduction

Fake news detection is a crucial task in today's digital age, where misinformation can spread rapidly. This project aims to classify news articles as real or fake using machine learning techniques. We use the TF-IDF vectorizer and the Passive Aggressive Classifier for this purpose.

## Dataset Description

The dataset used in this project consists of news articles with the following columns:
- `id`: Unique identifier for each article
- `title`: Title of the article
- `author`: Author of the article
- `text`: Full text of the article
- `label`: Label indicating whether the news is real or fake (1: fake, 0: real)

## Data Preprocessing

Before training the model, we need to preprocess the data. This involves handling missing values and converting the text data into numerical format using the TF-IDF vectorizer.

### Handling Missing Values

```python
data = data.dropna()
```
We drop any rows with missing values to ensure our dataset is clean and complete.

### Text Vectorization

```python
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
```
We use the `TfidfVectorizer` to convert the text data into numerical format. This helps the machine learning model understand the text.

## Model Building

We use the Passive Aggressive Classifier for this classification task. This model is known for its efficiency in binary classification tasks.

```python
pac = PassiveAggressiveClassifier(max_iter=50)
```
The Passive Aggressive Classifier is initialized with a maximum of 50 iterations.

## Model Training

We train the model using the training data.

```python
pac.fit(tfidf_train, y_train)
```
The `fit` method is used to train the model on the TF-IDF transformed training data and corresponding labels.

## Model Evaluation

After training, we evaluate the model's performance using accuracy and confusion matrix.

### Accuracy Score

```python
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')
```
We predict the labels for the test data and calculate the accuracy score.

### Confusion Matrix

```python
confusion_matrix(y_test, y_pred, labels=[0, 1])
```
The confusion matrix helps us understand the model's performance in detail by showing true positives, true negatives, false positives, and false negatives.

## Prediction

We can use the trained model to predict whether new articles are real or fake.

```python
new_text = ["Sample news article text"]
tfidf_new = tfidf_vectorizer.transform(new_text)
prediction = pac.predict(tfidf_new)
print(prediction)
```
We transform the new text data using the TF-IDF vectorizer and use the trained model to make predictions.

## Conclusion

This project demonstrates the use of machine learning for fake news detection. The Passive Aggressive Classifier, combined with the TF-IDF vectorizer, provides a robust solution for classifying news articles as real or fake.

---

This README provides a comprehensive overview of the fake news detection project, including detailed explanations of each step and code snippet. This should serve as a useful guide for anyone looking to understand or replicate the project.
