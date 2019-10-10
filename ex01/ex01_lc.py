#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#ML for NLP 2019
#Author: Debora Beuret

import numpy
import sklearn
import json
import numpy as np
import pandas as pd

filename1 = "tweets.json"
filename2 = "labels-train+dev.tsv"

infile1 = open(filename1, 'r')
infile2 = open(filename2, 'r')


"""def Sort(sub_li, n):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key=lambda x: x[n])
    return sub_li"""


mock_tweets_list = [[483885347374243841, 484023414781263872, 484026168300273664], ["اللهم أفرح قلبي وقلب من أحب وأغسل أحزاننا وهمومنا وأغفر ذنوبنا إنك على كل شيء قدير...",
"إضغط على منطقتك يتبين لك كم يتبقى من الوقت عن الآذان..\n\nhttp://t.co/2qdr9TEG1z", "اللَّهٌمَّ صَلِّ وَسَلِّمْ عَلىٰ نَبِيِّنَآ مُحَمَّدْ وَعَلَىٰ آلِھِہ وَصَحْبِھِہ أَجْمَعِينَ\n#غرد_لي_بالخير\nhttp://t.co/fN0Vvw0SZC"]]

tweets_df = pd.DataFrame(mock_tweets_list)
tweets_df = tweets_df.transpose()
tweets_df.columns = ["Tweet_ID", "Tweet"]

# The labels
train_labels_fp = "labels-train+dev.tsv"
test_labels_fp = "labels-test.tsv"

train_labels_df = pd.read_csv(train_labels_fp, sep='\t', names = ["Language", "Tweet_ID"])
test_labels_df = pd.read_csv(test_labels_fp, sep='\t', names = ["Language", "Tweet_ID"])

tweets_train = tweets_df.merge(train_labels_df, on='Tweet_ID', how='left')
tweets_train = tweets_train.drop('Tweet_ID', 1)


tweets_test = tweets_df.merge(test_labels_df, on='Tweet_ID', how='left')
tweets_test = tweets_test.drop('Tweet_ID', 1)

"""X_train = tweets_train.loc[:, "Tweet"].to_frame()
y_train = tweets_train.loc[:, "Language"].to_frame()"""
X_train = list(tweets_train["Tweet"])
y_train = list(tweets_train["Language"])


"""list_tweets = []
list_ids1 = []
for line in infile1:
    j_content = json.loads(line)
    list_ids1.append(int(j_content[0]))
    list_tweets.append(j_content[1])


list_lan = []
list_ids2 = []
for line in infile2:
    nl = line.split()
    list_lan.append(nl[0])
    list_ids2.append(int(nl[1]))


def_list = []

for el in list_ids1:
    if el in list_ids2:
        idx1 = list_ids1.index(el)
        idx2 = list_ids2.index(el)
        def_list.append([list_tweets[idx1], list_lan[idx2]])

print(len(list_ids1))
print(len(list_ids2))"""




from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

#This is our pipeline

#text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('nb_clf', MultinomialNB())])   #Example from Tutoring session

#could add features such as the word length, average tweet length, etc


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts tweet column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, tweet):
        """Helper code to compute average word length of a tweet"""
        print(tweet)
        print(type(tweet))
        return np.mean([len(word) for word in tweet.split()])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return map(self.average_word_length, df)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


#Deal with the Naive Bayes first. In the pipeline, the features used are n-grams, tfidf and average word length
## QUESTION: is featureunion needed or not?

pipeline_NB = Pipeline([
    ('feats', FeatureUnion([
        ('tfidf', TfidfVectorizer()), # can pass in either a pipeline
        ('ave', AverageWordLengthExtractor()), # or a transformer
        ('ngram', CountVectorizer(ngram_range=(1, 4), analyzer='word'))
    ])),
    ('clf1', MultinomialNB()) # classifier
])

grid_param_NB = {'clf1__alpha': [0.2, 0.6, 0.8, 1.0],
                 'clf1__fit_prior': [True, False]}  #'ngram__ngram_range': [(1, 1), (1, 2), (1, 4)]

gs_NB = GridSearchCV(pipeline_NB, grid_param_NB, cv=2, n_jobs=4, verbose=1)
gs_NB.fit(X_train, y_train)

#Stochastic gradient descent

pipeline_SGD = Pipeline([
    ('feats', FeatureUnion([
        ('tfidf', TfidfVectorizer()), # can pass in either a pipeline
        ('ave', AverageWordLengthExtractor()), # or a transformer
        ('ngram', CountVectorizer(ngram_range=(1, 4), analyzer='word'))
    ])),
    ('clf2', SGDClassifier())# classifier
])

grid_param_SGD = {'clf2__loss': ['hinge', 'log', 'modified huber'],
                  'clf2__penalty': ['none', 'l1', 'l2'],
                  'clf2__max_iter': [50, 100, 500, 1000]}


gs_SGD = GridSearchCV(pipeline_SGD, grid_param_SGD, cv=2, n_jobs=4, verbose=1)
gs_SGD.fit(X_train, y_train)





