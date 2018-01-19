#%%
import pyoo
import pandas as pd
desktop = pyoo.Desktop('localhost', 2002)
doc = desktop.create_spreadsheet()
sheet = doc.sheets[0]


#%%
        
data = [[3, 4], 
        [5, 6]]
data = pd.DataFrame(data)
shape = data.shape

start_row = 0
end_row = start_row + shape[0]
start_col = 0
end_col = start_col + shape[1]

sheet[start_row:end_row,start_col:end_col].values = data.values

#%%
df = pd.read_pickle('twitter_db.pickle')

for i in range(df.shape[0]):
    for j, value in enumerate(df.iloc[i].values):
        sheet[i,j].formula = value
#%%
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template
pd.set_option('display.max_colwidth', 1500)

vectorizer = TfidfVectorizer(stop_words='english',
                     binary=False,
                     max_df=0.95, 
                     min_df=0.011,
                     ngram_range=(1,2),
                     use_idf=False,
                     norm=None)

df = pd.DataFrame([])
df['description'] = ['This is a string1', 'This is a string2', 'This is a string3', 'This is a string4']
doc_vectors = vectorizer.fit_transform(df['description'])








