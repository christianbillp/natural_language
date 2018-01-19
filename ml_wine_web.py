# %% Imports
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template
pd.set_option('display.max_colwidth', 1500)

# %% Load data from file and process
df = pd.read_csv("datasets/winemag-data_first150k.csv", index_col=0)
df.columns = [name.lower() for name in df.columns.values]

query = '''A full-bodied and powerful wine with aromatic hints of blackberry and earthy tones.'''

query = '''Gewurztraminer's remarkable aroma – reminiscent of musk and lychee – tends to attract at first, then repel with its unrelenting attack on the senses.
I've toyed with the variety for about 40 years, since falling in and out of love, in one evening, with a particularly heady version from Alsace. The fascination returned recently with tastings of the 2014 and 2015 vintages from Mark Kirkby's Toppers Mountain vineyard.
The 2014 tasted so pure and delicate at the Winewise Championship; and a few days later the new-release 2015 showed similar class.
This is dry, intense gewurztraminer of the highest order. It's a wine to admire, but probably not drink much of.
'''
query = '''Austria's national white variety, gruner veltliner, now grows in several cool Australian regions, including Canberra and the Adelaide Hills.
The Clare-based Pike family source theirs from Lenswood, a particularly cool part of the Adelaide Hills which, like Clare, lies on the Mount Lofty Ranges.
The 2015 gruner tingles and pleases with its richly textured palate and tart melon-rind and pear-like flavours.
It finishes dry and refreshing with a distinctive spicy aftertaste.
'''

# %% Load data from file and process
df = pd.read_csv("datasets/IMDB-Movie-Data.csv", index_col=0)
df.columns = [name.lower() for name in df.columns.values]

query = '''A humble businessman with a buried past seeks justice when his 
daughter is killed in an act of terrorism. A cat-and-mouse conflict ensues 
with a government official, whose past may hold clues to the killers' 
identities.
'''

query = '''A group of misfits enter a Las Vegas dodgeball tournament in order 
to save their cherished local gym from the onslaught of a corporate health 
fitness chain.
'''

# %% Setup vectorizer to unigram vector representation
#    Vectorize query - (Optional: shape and feature names)
vectorizer = TfidfVectorizer(stop_words='english',
                     binary=False,
                     max_df=0.95, 
                     min_df=0.011,
                     ngram_range=(1,2),
                     use_idf=False,
                     norm=None)

doc_vectors = vectorizer.fit_transform(df['description'])
#query_vector = vectorizer.transform([query])
#df['similarity'] = cosine_similarity(query_vector, doc_vectors.toarray()).T
#print(doc_vectors.shape)
#print(vectorizer.get_feature_names())
print(f"Number of features: {len(vectorizer.get_feature_names())}")

# %% Create similarity function

def find_similar_descriptions(query, n_return_results=10):
    query_vector = vectorizer.transform([query])
    df['similarity'] = cosine_similarity(query_vector, doc_vectors.toarray()).T
    
    return df.sort_values('similarity', ascending=False).iloc[0:n_return_results]

#return_columns = ['similarity', 'description', 'country', 'price', 'designation', 'winery']

rdf = find_similar_descriptions(query)

# %% Web frontend
stylestring = """
<style>

table {
        font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
        border-collapse: collapse;
        width: 100%;
    }

td, th {
    border: 1px solid #ddd;
    padding: 8px;
}

tr:nth-child(even){background-color: #f2f2f2;}

tr:hover {background-color: #abc;}

th {
    padding-top: 12px;
    padding-bottom: 12px;
    text-align: left;
    background-color: #4CAF50;
    color: white;
}
</style>
"""


app = Flask(__name__)
@app.route('/')
def render_frontend():
    return render_template('wine_frontend.html')

@app.route('/', methods=['POST'])
def return_results():
    return_columns = ['title', 'genre', 'description', 'metascore', 'similarity']
    text = request.form['text']
    result = find_similar_descriptions(text)[return_columns]
#    return render_template('wine.html')
    return stylestring + result.to_html()

if __name__ == "__main__":
    app.run()   