import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load the csv file
df_reviews = pd.read_csv('IMDB Dataset.csv')

# Rename sentiment column to label
df_reviews = df_reviews.rename(columns={'sentiment': 'label'})

# Replace positive and negative to ones and zeros
df_reviews['label'] = df_reviews['label'].replace(
    {'positive': 1, 'negative': 0})

# Replace <br /><br /> to empty string
df_reviews['review'] = df_reviews['review'].str.replace("<br /><br />", '')

# Replace \ to empty string
df_reviews['review'] = df_reviews['review'].str.replace("\"", '')

# Separate data and assign review and label columns to two arrays
x = np.array(df_reviews['review'])
y = np.array(df_reviews['label'])


# Vectorize textual values to numerical values
cv = CountVectorizer(stop_words="english")
x = cv.fit_transform(x)


# Split the data into train 70% and test 30% datasets
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, random_state=42,  test_size=0.3)

# Naive Bayes
nb_model = MultinomialNB()

# Fit the model
nb_model = nb_model.fit(X_train, Y_train)

# Create a joblib file
joblib.dump(nb_model, 'model_joblib')
