import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import lstsq
from sklearn.metrics import mean_squared_error

arts = pd.read_pickle('articles.pkl')
arts_content = arts['content']

vectorizer = CountVectorizer(stop_words= 'english',max_features=5000)

X = vectorizer.fit_transform(arts_content)

features = vectorizer.get_feature_names()

class NMF(object):

    def __init__(self,V,k,its):
        self.V = V
        self.k = k
        self.its = its
        self.W = np.random.random((V.shape[0],k))*V.max()
        self.H = np.ones((k,V.shape[1]))

    def fit(self):
        for i in np.arange(1,self.its+1):
            if i%2 == 1:
                self.H = lstsq(self.V, self.W)[0] # hold W constant, solve for H
                self.H[self.H < 0] = 0
            else:
                self.W = lstsq(self.V.T, self.H)[0] # hold H constant, solve for W
                self.W[self.W < 0] = 0
        return self.W, self.H.T

    def score(self):
        self.fit()
        return mean_squared_error(self.V, np.dot(self.W,self.H.T))
