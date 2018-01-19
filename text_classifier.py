#%%
%matplotlib inline

import sqlite3
import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer


# %% Extract data from database

class TextClassifier():
    
    def __init__(self, dataframe):
        self.df = dataframe
        
    def classify(self):
        self.df["Sentiment"] = self.df["Score"].apply(lambda score: "positive" if score > 3 else "negative")
        self.df["Usefulness"] = (self.df["VotesHelpful"]/self.df["VotesTotal"]).apply(lambda n: "useful" if n > 0.8 else "useless")

        
    def test():
        
        
if __name__ == '__main__':
    con = sqlite3.connect('datasets/database.sqlite')
    messages = pd.read_sql_query("""
    SELECT
      Score,
      Summary,
      HelpfulnessNumerator as VotesHelpful,
      HelpfulnessDenominator as VotesTotal
    FROM Reviews
    WHERE Score != 3""", con)


tc = TextClassifier(messages)
tc.classify()
        
