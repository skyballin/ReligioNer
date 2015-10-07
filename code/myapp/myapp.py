from flask import Flask, request, render_template
import re
import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

with open('../../data/merged.pkl', 'r') as f:
    merged = pickle.load(f)
with open('../../data/vectorizer.pkl', 'r') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/examples')
def examples():
    return render_template('examples.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/text_similarity', methods=['POST'])
def text_similarity():
    user_orig = str(request.form['user_input'])
    lmtzr = WordNetLemmatizer()
    user_input = re.sub("[^a-zA-Z]", " ", user_orig)
    user_input = lmtzr.lemmatize(user_input).lower()

    user_vect = vectorizer.transform([user_input])
    
    similarity_list = []
    for vectors in merged['Vectors']:
        similarity_list.append(cosine_similarity(user_vect, vectors))
    merged['user_similarity'] = similarity_list
    
    max_vectors = []
    for source in np.unique(merged['Source']):
        sourcedf = merged[merged['Source'] == source]
        max_vector = merged[merged['Source'] == source].ix[np.argmax(sourcedf['user_similarity'])]
        max_vectors.append(max_vector)
    
    source0 = max_vectors[0]['Source'] # Mormon
    source1 = max_vectors[1]['Source'] # Sutras
    source2 = max_vectors[2]['Source'] # Gita
    source3 = max_vectors[3]['Source'] # Koran
    source4 = max_vectors[4]['Source'] # New Testament
    source5 = max_vectors[5]['Source'] # Torah

    book0 = max_vectors[0]['Book']
    book1 = max_vectors[1]['Book']
    book2 = max_vectors[2]['Book']
    book3 = max_vectors[3]['Book']
    book4 = max_vectors[4]['Book']
    book5 = max_vectors[5]['Book']

    chapter0 = str(max_vectors[0]['Chapter'])
    chapter1 = str(max_vectors[1]['Chapter'])
    chapter2 = str(max_vectors[2]['Chapter'])
    chapter3 = str(max_vectors[3]['Chapter'])
    chapter4 = str(max_vectors[4]['Chapter'])
    chapter5 = str(max_vectors[5]['Chapter'])
    
    verse0 = str(max_vectors[0]['Verse'])
    verse1 = str(max_vectors[1]['Verse'])
    verse2 = str(max_vectors[2]['Verse'])
    verse3 = str(max_vectors[3]['Verse'])
    verse4 = str(max_vectors[4]['Verse'])
    verse5 = str(max_vectors[5]['Verse'])

    text0 = max_vectors[0]['Original Text']
    text1 = max_vectors[1]['Original Text']
    text2 = max_vectors[2]['Original Text']
    text3 = max_vectors[3]['Original Text']
    text4 = max_vectors[4]['Original Text']
    text5 = max_vectors[5]['Original Text']

    if len(text0) > 150:
        text0 = text0[:150]+"..."
    if len(text1) > 150:
        text1 = text1[:150]+"..."
    if len(text2) > 150:
        text2 = text2[:150]+"..."
    if len(text3) > 150:
        text3 = text3[:150]+"..."
    if len(text4) > 150:
        text4 = text4[:150]+"..."
    if len(text5) > 150:
        text5 = text5[:150]+"..."

    cos0 = str(round(max_vectors[0]['user_similarity'][0][0]*100,2)) + "%"
    cos1 = str(round(max_vectors[1]['user_similarity'][0][0]*100,2)) + "%"
    cos2 = str(round(max_vectors[2]['user_similarity'][0][0]*100,2)) + "%"
    cos3 = str(round(max_vectors[3]['user_similarity'][0][0]*100,2)) + "%"
    cos4 = str(round(max_vectors[4]['user_similarity'][0][0]*100,2)) + "%"
    cos5 = str(round(max_vectors[5]['user_similarity'][0][0]*100,2)) + "%"

    output_orig = "Original Query: " + user_orig + "<br>"
    
    if cos0 == "0.0%":
        output0 = "There is nothing in the Book of Mormon resembling this phrase."
    else:
        output0 = "Source: " + source0 + "<br>" + "Book: " + book0 + "<br>" + "Chapter: " + chapter0 + "<br>" + "Verse: " + verse0 + "<br>" + "Text: <br>" + text0 + "<br>" + "Similarity: " + cos0
    if cos1 == "0.0%":
        output1 = "There is nothing in the Buddhist Sutras resembling this phrase"
    else:
        output1 = "Source: " + source1 + "<br>" + "Book: " + book1 + "<br>" + "Chapter: " + chapter1 + "<br>" + "Verse: " + verse1 + "<br>" + "Text: <br>" + text1 + "<br>" + "Similarity: " + cos1
    if cos2 == "0.0%":
        output2 = "There is nothing in the Bhagavad Gita resembling this phrase"
    else:
        output2 = "Source: " + source2 + "<br>" + "Book: " + book2 + "<br>" + "Chapter: " + chapter2 + "<br>" + "Verse: " + verse2 + "<br>" + "Text: <br>" + text2 + "<br>" + "Similarity: " + cos2
    if cos3 == "0.0%":
        output3 = "There is nothing in the Koran resembling this phrase"
    else:
        output3 = "Source: " + source3 + "<br>" + "Book: " + book3 + "<br>" + "Chapter: " + chapter3 + "<br>" + "Verse: " + verse3 + "<br>" + "Text: <br>" + text3 + "<br>" + "Similarity: " + cos3
    if cos4 == "0.0%":
        output4 = "There is nothing in the the New Testament resembling this phrase"
    else:
        output4 = "Source: " + source4 + "<br>" + "Book: " + book4 + "<br>" + "Chapter: " + chapter4 + "<br>" + "Verse: " + verse4 + "<br>" + "Text: <br>" + text4 + "<br>" + "Similarity: " + cos4
    if cos5 == "0.0%":
        output5 = "There is nothing in the the Torah resembling this phrase"
    else:
        output5 = "Source: " + source5 + "<br>" + "Book: " + book5 + "<br>" + "Chapter: " + chapter5 + "<br>" + "Verse: " + verse5 + "<br>" + "Text: <br>" + text5 + "<br>" + "Similarity: " + cos5

    if cos0 == '0.0%' and cos1 == '0.0%' and cos2 == '0.0%' and cos3 == '0.0%' and cos4 == '0.0%' and cos5 == '0.0%':
        out = [output0]
        for i in xrange(6): 
            out.append("There is no religious context for your query")
        return render_template('text_similarity.html', data = out)
        
    return render_template('text_similarity.html', data = [output_orig, output0, output1, output2, output3, output4, output5])

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=80, debug=False)

