# -*- coding: utf-8 -*-
"""

@author: Zachary Lanz
    
1)	Run the program up until the movie reviews documents.
2)	Run the “documents = movies_reviews” along with the first preprocessing/baseline.
3)	Run the second preprocessing.
4)	DO NOT RUN THE TEST SET FOR ASCII STRIP ACCENT REMOVAL
5)	Run the third preprocessing. 
6)	All done!


"""

#import nltk
#import random
#import collections

#import movie review corpus
from nltk.corpus import movie_reviews

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# A function to display the top terms in each topic after the model has been
# created.
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))

#set the number of features to use and the number of topics for each preprocessing
num_features = 1000
num_topics = 5
num_topics2 = 5
num_topics3 = 6


# set the number of top words we want to display for each topic
num_top_words = 8
num_top_words2 = 6
num_top_words3 = 1

# display the words associated with the topics
#display_topics(lda, tf_feature_names, num_top_words)


#Do it all again with movie reviews#


# STOP FIRST RUN HERE PER INSTRUCTIONS IN WRITE-UP #

# Start Run With First Preprocessing/baseline Establishment #
documents = [movie_reviews.raw(fileid)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

print ("Baseline/Preprocessing 1:")
#preprocess a baseline analyzing 1000 features
tf_vectorizer1 = CountVectorizer(max_df=1, min_df=1, max_features=num_features, ngram_range=(2,2))
tf = tf_vectorizer1.fit_transform(documents)
tf_feature_names = tf_vectorizer1.get_feature_names()

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
display_topics(lda, tf_feature_names, num_top_words)

print ("Preprocessing 2:")
#expand from baseline to include the removal of accents through unicode
tf_vectorizer2 = CountVectorizer(max_df=.7, min_df=1, max_features=num_features, stop_words='english', ngram_range=(2,2))
tf = tf_vectorizer2.fit_transform(documents)
tf_feature_names2 = tf_vectorizer2.get_feature_names()

lda = LatentDirichletAllocation(n_components=num_topics2, max_iter=35, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
display_topics(lda, tf_feature_names2, num_top_words2)


##### test documents to see what ascii strip accents is doing######
# ignore for full test run #

ascii_documents = ["they duck around the slow_duck",
              "the frog likes to watch a frog",
              "the monkey climbs a big tree",
              "the bird ate 1: frog",
              "fish---likes to fish",
              "she likes going into the zoo",
              "he climbs on your favorite horse",
              "he makes apple and tomato pizza",
              "hot pizza is good",
              "small-children eat candy",
              "he drinks ten $%drinks",
              "help me eat 3: pizzas",
              "your balloon went up",
              "balloon travel &&makes travel cool",
              "his car is-a fast car",
              "this airplane: flies into your school",
              "she read a book to book travel",
              "dark clothes are not good after dark",
              "we wear new red--- shoes",
              "what a shiny black hat!"]

print ("Testing Documents for strip accents ascii ONLY:")

tf_vectorizer4 = CountVectorizer(max_df=.7, min_df=1, max_features=num_features, strip_accents='ascii')
tf = tf_vectorizer4.fit_transform(ascii_documents)
asciitestdocuments = tf_vectorizer4.get_feature_names()

lda = LatentDirichletAllocation(n_components=num_topics3, max_iter=50, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
display_topics(lda, asciitestdocuments, num_top_words3)

#### end of test set for ascii ######



print ("Preprocessing 3:")

tf_vectorizer3 = CountVectorizer(max_df=.7, min_df=1, max_features=num_features, strip_accents='ascii', stop_words='english', ngram_range=(2,2))
tf = tf_vectorizer3.fit_transform(documents)
tf_feature_names3 = tf_vectorizer3.get_feature_names()

lda = LatentDirichletAllocation(n_components=num_topics3, max_iter=50, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
display_topics(lda, tf_feature_names3, num_top_words3)
