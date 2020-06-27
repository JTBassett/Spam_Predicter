import nltk
import csv
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
#TfidfVectorizer is another good option to consider, but not as easy to then match on the app.py side
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle


""" Read in and clean the text data """

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.csv", sep='\t')
data.columns = ['label', 'body_text']

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

count_vect = CountVectorizer(analyzer=clean_text)
X_counts = count_vect.fit_transform(data['body_text'])

#Keep names for when looking at importance features
main_df = pd.DataFrame(X_counts.toarray())
main_df.columns = count_vect.get_feature_names()

X_features = pd.concat([data['body_len'], data['punct%'], main_df], axis=1)

#Create csv of features for reference in app.py
with open('features.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in X_features:
        writer.writerow([val])   
    print("Should have created feature file")

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X_features, data['label'], test_size=0.2)

#Now fit the model

#max_depth default is None, or how many needed
#Use GradientBoostingClassifer as compared to RandomForest because of greater Recall
gb = GradientBoostingClassifier(n_estimators=150, max_depth=11)
gb_model = gb.fit(X_train, y_train)

print("We have a model working")

pickle.dump(gb_model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

