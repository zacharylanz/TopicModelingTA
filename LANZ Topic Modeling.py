# -*- coding: utf-8 -*-
"""

@author: Zachary Lanz

Instructions for Dr. Young:
    
1)	Run the program up until the movie reviews documents.
2)	Run the “documents = movies_reviews” along with the first preprocessing/baseline.
3)	Run the second preprocessing.
4)	DO NOT RUN THE TEST SET FOR ASCII STRIP ACCENT REMOVAL
5)	Run the third preprocessing. 
6)	All done!


"""



#imported just incase any was necessary 
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



"""
Challenge: re-run with different 'preprocessing' choices for CountVectorizer
and different numbers of iterations.


Here are some potential arguments for CountVectorizer from http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

max_iter : how many cycles of moving around the features do you want to run. More iterations may lead to more refined results.

strip_accents : {‘ascii’, ‘unicode’, None}

    Remove accents and perform other character normalization during the preprocessing step. ‘ascii’ is a fast method that only works on characters that have an direct ASCII mapping. ‘unicode’ is a slightly slower method that works on any characters. None (default) does nothing.

    Both ‘ascii’ and ‘unicode’ use NFKD normalization from unicodedata.normalize.


lowercase : boolean, True by default

    Convert all characters to lowercase before tokenizing.



stop_words : string {‘english’}, list, or None (default)

    If ‘english’, a built-in stop word list for English is used. There are several known issues with ‘english’ and you should consider an alternative (see Using stop words).

    If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'.

    If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.

    token_pattern : string

    Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'. The default regexp select tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).


ngram_range : tuple (min_n, max_n)

    The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.



max_df : float in range [0.0, 1.0] or int, default=1.0

    When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.


min_df : float in range [0.0, 1.0] or int, default=1

    When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.


max_features : int or None, default=None

    If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

    This parameter is ignored if vocabulary is not None.


"""

