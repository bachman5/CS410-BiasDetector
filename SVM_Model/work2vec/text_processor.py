from collections import defaultdict

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from SVM_Model.work2vec.Tokenizer import Tokenizer as Toki


# Custom transformer using spaCy
class TextNormalizer(TransformerMixin):
    def transform(self, X, **transform_params):
        clean_sents = [Toki.normalize_text(line) for line in X]
        list_word = [stt.split() for stt in clean_sents]
        return list_word

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# Credit http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class MeanEmbeddingVectorizer(object):
    word2vec = None
    dim = None

    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            #np.mean([self.word2vec.query(w) for w in words if w in self.word2vec]
            #        or [np.zeros(self.dim)], axis=0)
            np.mean([self.word2vec.query(w) for w in words], axis=0)
            for words in X
        ])


# Credit http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = word2vec.dim

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec.query(w) * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])