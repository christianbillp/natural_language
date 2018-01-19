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
con = sqlite3.connect('datasets/database.sqlite')
#pd.read_sql_query("SELECT * FROM Reviews LIMIT 3", con)

messages = pd.read_sql_query("""
SELECT
  Score,
  Summary,
  HelpfulnessNumerator as VotesHelpful,
  HelpfulnessDenominator as VotesTotal
FROM Reviews
WHERE Score != 3""", con)

# %% Create sentiment from score
messages["Sentiment"] = messages["Score"].apply(lambda score: "positive" if score > 3 else "negative")
messages["Usefulness"] = (messages["VotesHelpful"]/messages["VotesTotal"]).apply(lambda n: "useful" if n > 0.8 else "useless")

# %% Clean summaries and split dataset into training and test set
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    #sentence = " ".join(nltk.word_tokenize(sentence))
    return sentence

messages["Summary_Clean"] = messages["Summary"].apply(cleanup)
train, test = train_test_split(messages, test_size=0.2)
print("%d items in training data, %d in test data" % (len(train), len(test)))

# %% Analysis on prepared dataframe
# note: Names "Summary_Clean" and "Sentiment" may need to be changed

tfidf_transformer = TfidfTransformer()

count_vect = CountVectorizer(min_df = 1, ngram_range = (1, 4))
X_train_counts = count_vect.fit_transform(train["Summary_Clean"])
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_new_counts = count_vect.transform(test["Summary_Clean"])

X_test_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = dict()
y_train = train["Sentiment"]
y_test = test["Sentiment"]

#%% Applying Multinomial Naïve Bayes learning method
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, y_train)
prediction['Multinomial'] = model.predict(X_test_tfidf)

#%% Applying Bernoulli Naïve Bayes learning method
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train_tfidf, y_train)
prediction['Bernoulli'] = model.predict(X_test_tfidf)

#%% Applying Logistic regression learning method
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
logreg_result = logreg.fit(X_train_tfidf, y_train)
prediction['Logistic'] = logreg.predict(X_test_tfidf)

#%% Compare results from different learning methods
def formatt(x):
    if x == 'negative':
        return 0
    return 1

vfunc = np.vectorize(formatt)

#cmp = 0    # Related to plotting
colors = ['b', 'g', 'y', 'm', 'k']
results = {}
cmp = 0
# Check accuracy for each learning method
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    results[model] = roc_auc

#print(results)

    # Uncomment for plot

    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%
def test_sample(model, sample):
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    prob = model.predict_proba(sample_tfidf)[0]
    print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))

def get_sentiment(sample):
    model = logreg
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    prob = model.predict_proba(sample_tfidf)[0]
    
    return result.upper()
#    return prob

test_sample(logreg, 'The food was delicious, it smelled great and the taste was awesome')
get_sentiment('This place is terrible! Everything tasted disgusting...')
