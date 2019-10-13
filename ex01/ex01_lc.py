"""
ML4NLP - Exercise 01

Linear classifiers

Authors: David Bielik, Debora Beuret
"""

import json
import numpy as np
import pandas as pd
from math import floor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Constants
# File paths
TWEETS_FP = "tweets.json"
TRAIN_DEV_FP = "labels-train+dev.tsv"
TEST_FP = "labels-test.tsv"

# Column names
COL_ID = 'ID'
COL_TWEET = 'Tweet'
COL_LABEL = 'Label'

# minimum number of instances that we require to be present in the training set
# for a given language to be included in fitting of the model
MIN_NR_OCCURENCES = 10

# unknown class name
CLASS_UNK = 'unknown'


def get_tweets():
    """Return a dataframe of tweets"""
    tweets = []
    with open(TWEETS_FP, 'r') as tweets_fh:
        for line in tweets_fh:
            j_content = json.loads(line)
            tweets.append(j_content)

    # make a dataframe out of it
    tweets = pd.DataFrame(tweets, columns=[COL_ID, COL_TWEET])
    return tweets

def get_train_labels():
    """Return a dataframe of train_dev labels"""
    train_dev_labels = pd.read_csv(
        TRAIN_DEV_FP,
        sep='\t',
        header=None,
        names=[COL_LABEL, COL_ID]
    )
    # remove whitespace from labels (e.g. "ar  " should be equal to "ar")
    train_dev_labels.Label = train_dev_labels.Label.str.strip()

    # deal with class imbalance in the train set
    lang_occurence = train_dev_labels.groupby(COL_LABEL).size()
    balanced_languages = lang_occurence.where(
        lang_occurence >= MIN_NR_OCCURENCES
    ).dropna().index.values
    balanced_labels = train_dev_labels.Label.isin(balanced_languages)


    # Option 1 - replace rows that are labelled with an imbalanced language
    # ~ is element-wise logical not
    train_dev_labels.loc[~balanced_labels, COL_LABEL] = CLASS_UNK

    # Option 2 - keep the rows that are labelled with a balanced language
    # train_dev_labels = train_dev_labels[balanced_labels]
    return train_dev_labels

def get_test_labels():
    """Return a dataframe of test labels"""
    return pd.read_csv(
        TEST_FP,
        sep='\t',
        header=None,
        names=[COL_LABEL, COL_ID]
    )

def create_sets(tweets, train_dev_labels, test_labels, use_dev=True):
    """Return a tuple of dataframes comprising three main data sets"""
    # to allow for merge, need the same type
    tweets[COL_ID] = tweets[COL_ID].astype(int)

    # Merge by ID
    train_dev_data = pd.merge(tweets, train_dev_labels, on=COL_ID)
    test_data = pd.merge(tweets, test_labels, on=COL_ID)

    # Util function
    def drop_n_shuffle(data):
        data_no_na = data.dropna().copy()
        return data_no_na.sample(frac=1)

    frac = 1
    if use_dev:
        frac = 0.9

    train_dev_data_prepared = drop_n_shuffle(
        train_dev_data
    ).reset_index(drop=True)
    # take frac of the data (e.g. 90%), reshuffle
    train_set = train_dev_data_prepared.sample(frac=frac, random_state=0)
    # take the remaining data
    dev_set = train_dev_data_prepared.drop(train_set.index)
    test_set = drop_n_shuffle(test_data)

    # drop the ID columns, not needed anymore
    train = train_set.drop(COL_ID, axis=1)
    dev = dev_set.drop(COL_ID, axis=1)
    test = test_set.drop(COL_ID, axis=1)

    return train, dev, test

# Average word length extractor, inspired by:
# https://michelleful.github.io/code-blog/2015/06/20/pipelines/)
class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts tweet column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, tweet):
        """Helper code to compute average word length of a tweet"""
        return np.mean([len(word) for word in tweet.split()])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        # the result of the transform needs to be a 2d array a.k.a. dataframe
        # https://stackoverflow.com/a/50713209
        result = df.apply(self.average_word_length).to_frame()
        return result

    def fit(self, df, y=None):
        """Returns `self` unless something happens in train and test"""
        return self


def train_MNB(X_train, y_train):
    """Return the Multinomial Naïve Bayes model trained on the parameters"""
    multinomial_NB = Pipeline([
        ('features', FeatureUnion([
            ('ngram_tfidf', Pipeline([
                ('ngram', CountVectorizer(analyzer='word', ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer())
            ])),
            ('ave_scaled', Pipeline([
                ('ave', AverageWordLengthExtractor()),
                ('scale', MinMaxScaler())
            ]))
        ])),
        ('nb_clf', MultinomialNB(fit_prior=False, alpha=0.1)) # classifier
    ])
    # train
    multinomial_NB.fit(X_train, y_train)
    return multinomial_NB

def train_SGD(X_train, y_train):
    """Return the Stochastic Gradient Descent model trained on the parameters"""
    SGD = Pipeline([
        ('feats', FeatureUnion([
            ('ngram_tfidf', Pipeline([
                ('ngram', CountVectorizer(ngram_range=(1, 2), analyzer='word')),
                ('tfidf', TfidfTransformer()),
            ])),
            ('ave_scaled', Pipeline([
                ('ave', AverageWordLengthExtractor()),
                ('scale', MinMaxScaler())
            ]))
        ])),
        ('SGD_clf', SGDClassifier(loss='hinge', max_iter=300, penalty=None))
    ])

    # train
    SGD.fit(X_train, y_train)
    return SGD

def preprocess():
    """Return the three main sets used in the pipelines"""
    tweets = get_tweets()
    train_labels = get_train_labels()
    test_labels = get_test_labels()
    return create_sets(tweets, train_labels, test_labels, use_dev=False)

def main():
    # we don't need the dev set here
    train, _, test = preprocess()

    X_train = train.Tweet
    y_train = train.Label
    X_test = test.Tweet
    y_test = test.Label

    classifiers = {'SGD': train_SGD,
                   'MNB': train_MNB}

    for n, fn in classifiers.items():
        print("Fitting " + n + " classifier.")
        res = fn(X_train, y_train)
        y_predicted = res.predict(X_test)
        accuracy = accuracy_score(y_test, y_predicted) * 100
        print(n + " accuracy: " + "%.2f" % accuracy + "%")


if __name__ == "__main__":
    main()
