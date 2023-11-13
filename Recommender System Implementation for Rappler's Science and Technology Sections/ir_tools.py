import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sqlite3
from collections import Counter

def to_bow(docs):
    """Return a DataFrame of bag-of-words representations of a list of docs"""
    bow = {}
    for index, doc in enumerate(docs):
        for word, count in Counter(re.findall(r'\b[a-zA-Z]+\b', doc)).most_common():
            bow.setdefault(word, {i: 0 for i in range(len(docs))})
            bow[word].update({index: count})
    df = pd.DataFrame(bow)
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def lpnorm(vec1, vec2, p=2):
    """Compute the L_p-norm distance between vec1 and vec2

    If `vec1` and `vec2` are same-sized matrices, an ndarray of the L_p-norm 
    of corresponding rows will be returned instead.

    Parameters
    ----------
    vec1 : ndarray
        First vector
    vec2 : ndarray
        Second vector
    p : int or float, optional
        Order of L_p norm; the `p` in L_p norm

    Returns
    -------
    float
        L_p norm distance of `vec1` and `vec2`
    """
    a = len(vec1.shape) - 1
    return np.sum(np.abs((vec1 - vec2))**p, axis=a)**(1/p)


def cossim(vec1, vec2):
    """Compute cosine similarity between vec1 and vec2

    If `vec1` and `vec2` are same-sized matrices, an ndarray of the cosine 
    similarity of corresponding rows will be returned instead.

    Parameters
    ----------
    vec1 : ndarray
        First vector
    vec2 : ndarray
        Second vector

    Returns
    -------
    float
        cosine similarity of `vec1` and `vec2`
    """
    a = len(vec1.shape) - 1
    return np.sum(vec1*vec2, axis=a)/(np.sqrt((vec1**2).sum(axis=a)) * np.sqrt((vec2**2).sum(axis=a)))


def dcos(vec1, vec2):
    """Compute cosine distance between vec1 and vec2

    If `vec1` and `vec2` are same-sized matrices, an ndarray of the cosine 
    distance of corresponding rows will be returned instead.

    Parameters
    ----------
    vec1 : ndarray
        First vector
    vec2 : ndarray
        Second vector

    Returns
    -------
    float
        cosine distance of `vec1` and `vec2`
    """
    return 1 - cossim(vec1, vec2)


def nearest_k(query, objects, k, dist):
    """Return the indices to objects most similar to query

    Parameters
    ----------
    query : ndarray
        query object represented in the same form vector representation as the
        objects
    objects : ndarray
        vector-represented objects in the database; rows correspond to 
        objects, columns correspond to features
    k : int
        number of most similar objects to return
    dist : function
        accepts two ndarrays as parameters then returns their distance

    Returns
    -------
    ndarray
        Indices to the most similar objects in the database
    """
    return np.argsort(dist(np.repeat([query], len(objects), axis=0), objects))[:k]


class Vectorizer:
    def __init__(self):
        self.index_word = {}
        self.word_index = {}

    def build_mappings(self, docs):
        """Initialize word-index mappings

        Parameter
        ---------
        docs : sequence of str
            Corpus to build mappings for
        """
        unique_words = sorted(set(" ".join(
            [" ".join(re.findall(r'\b[a-zA-Z]+\b', doc)) for doc in docs]).split()))
        self.index_word = {num: word for num, word in enumerate(unique_words)}
        self.word_index = {word: num for num, word in enumerate(unique_words)}

    def vectorize(self, doc):
        """Return the BoW vector representation of doc

        Parameters
        ----------
        doc : str
            Text to compute the vector representation of

        Returns
        -------
        vec : ndarray
            BoW vector representation of doc
        """
        doc_words = Counter(re.findall(r'\b[a-zA-Z]+\b', doc)).most_common()
        vec = np.zeros(len(self.index_word))
        for word, count in doc_words:
            if word in self.word_index.keys():
                vec[self.word_index[word]] = count
        return vec


class TFIDF:
    def __init__(self, df):
        """Store the idf of each column"""
        self.idf = np.log(len(df) / np.sum(df > 0))

    def tfidf(self, values):
        """Standard values per column"""
        return values * self.idf


def normalize1(values):
    """Calculate the L1-norm of a vector or matrix of values."""
    values = np.asarray(values)
    a = len(values.shape) - 1
    if a > 0:
        return values / values.sum(axis=a)[:, np.newaxis]
    else:
        return values / values.sum(axis=a)


def normalize2(values):
    """Calculate the L2-norm of a vector or matrix of values."""
    values = np.asarray(values)
    a = len(values.shape) - 1
    if a > 0:
        return values / np.linalg.norm(values, axis=a)[:, np.newaxis]
    else:
        return values / np.linalg.norm(values, axis=a)
    
    
def get_confusion(actual, results, all_labels):
    """Calculate the confusion matrix of prediction results to actual value"""
    tp = len([i for i in results if all_labels[i] == actual])
    fp = len([i for i in results if all_labels[i] != actual])
    fn = len([i for i in np.delete(all_labels, results)
              if i == actual])
    tn = len([i for i in np.delete(all_labels, results)
              if i != actual])
    return pd.DataFrame([[tp, fp], [fn, tn]],
                        columns=['relevant', 'irrelevant'],
                        index=['relevant', 'irrelevant'])


def precision(confusion):
    """Calculate the precision of a confusion matrix."""
    tp = confusion.iloc[0, 0]
    fp = confusion.iloc[0, 1]
    return tp/(tp+fp)


def recall(confusion):
    """Calculate the recall of a confusion matrix."""
    tp = confusion.iloc[0, 0]
    fn = confusion.iloc[1, 0]
    return tp/(tp+fn)


def f_measure(precision, recall, beta=1):
    """Calculate the f-measure of a confusion matrix."""
    return (1+beta**2) * ((precision*recall)/((beta**2*precision) + recall))


def pr_curve(query, objects, dist, actual, all_labels):
    all_labels = np.asarray(all_labels)
    results = nearest_k(query, objects, len(all_labels), dist)
    rs = (all_labels[results] == actual).cumsum()
    N = (all_labels == actual).sum()
    precisions = rs / np.arange(1, len(rs)+1)
    recalls = rs / N
    recalls = [0] + recalls.tolist()
    precisions = [1] + precisions.tolist()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.step(recalls, precisions, where='post')
    ax.fill_between(recalls, precisions, step='post', alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    return ax


def auc_pr(query, objects, dist, actual, all_labels):
    all_labels = np.asarray(all_labels)
    results = nearest_k(query, objects, len(all_labels), dist)
    rs = (all_labels[results] == actual).cumsum()
    N = (all_labels == actual).sum()
    precisions = rs / np.arange(1, len(rs)+1)
    recalls = rs / N
    recalls = [0] + recalls.tolist()
    precisions = [1] + precisions.tolist()
    return np.trapz(precisions, recalls)