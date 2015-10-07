import pandas as pd
import numpy as np
import cPickle as pickle
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.wordnet import WordNetLemmatizer
from book import Book


class text_similarity():
    
    def __init__(self, booklist):
        self.booklist = booklist
        self.alltext = ''
        self.merged = pd.DataFrame(columns = ['Book', 'Chapter', 'Verse', 'Original Text', 'Formatted Text'])
        self.booklistnames = []
        for i, book in enumerate(booklist):
            self.alltext += book.all_text
            self.merged = pd.merge(self.merged, book.df, how = 'outer')
            self.booklistnames = np.append(self.booklistnames, book.unique_books)
        self.vocabulary = self.alltext.split()
        self.vocabulary = [word for word in self.vocabulary if len(word) > 2]
        self.cosinedf = pd.DataFrame(columns=self.booklistnames, index=self.booklistnames)
        self.merged['Source'] = '' 
        sources = []
        for i, book in enumerate(self.merged['Book']):
            for books in self.booklist:
                if book in books.unique_books:
                    sources.append(books.name)
        self.merged['Source'] = sources
        self.vect = TfidfVectorizer(stop_words='english')
        self.vect.fit_transform(self.vocabulary)
        self.vectorize()
        self.cosine()
    
    def vectorize(self):
        self.tfidf_df = pd.DataFrame(columns= ['Book', 'Vector'])
        self.tfidf_df['Book'] = self.booklistnames
        
        for i, book in enumerate(self.booklistnames):
            joined = " ".join(self.merged[self.merged['Book'] == book]['Formatted Text'].values)
            self.tfidf_df.iloc[i, 1] = self.vect.transform([joined])
        
        vectors = []
        for i, line in enumerate(self.merged['Formatted Text']):
            vectors.append(self.vect.transform([line]))
            if i % 10 == 0:
                print i
        self.merged['Vectors'] = vectors
        print "vectorization complete"

    def cosine(self):
        self.cosinedf['Source'] = ''
        sources = []
        
        for i, book in enumerate(self.cosinedf):
            for books in self.booklist:
                if book in books.unique_books:
                    sources.append(books.name)
        self.cosinedf['Source'] = sources

        for i, book1 in enumerate(self.booklistnames):
            for j, book2 in enumerate(self.booklistnames):
                if book1 == book2:
                    self.cosinedf[book1][book2] = 1.
                elif i<j:
                    self.cosinedf[book1][book2] = cosine_similarity(self.tfidf_df[self.tfidf_df['Book'] == book1]['Vector'].values[0], 
                                  self.tfidf_df[self.tfidf_df['Book'] == book2]['Vector'].values[0])[0][0]
        print "cosine similarity complete"    

if __name__ == "__main__":
    with open('../data/koran.pkl', 'r') as f:
        koran = pickle.load(f)
    with open('../data/gita.pkl', 'r') as f:
        gita = pickle.load(f)
    with open('../data/mormon.pkl', 'r') as f:
        mormon = pickle.load(f)
    with open('../data/sutras.pkl', 'r') as f:
        sutras = pickle.load(f)
    with open('../data/torah.pkl', 'r') as f:
        torah = pickle.load(f)
    with open('../data/newtestament.pkl', 'r') as f:
        newtestament = pickle.load(f)
    
    booklist = [koran, gita, mormon, sutras, torah, newtestament]
    output = text_similarity(booklist)

    with open('../data/vectorizer.pkl', 'wb') as f:
        pickle.dump(output.vect, f)
    print "vector pickle complete"
    
    with open('../data/merged.pkl', 'wb') as f:
        pickle.dump(output.merged, f)
    print "merged dataframe pickle complete"

    with open('../data/cosine.pkl', 'wb') as f:
        pickle.dump(output.cosinedf, f)
    print "cosine dataframe pickle complete"

    print "ALL IS GOOD IN THE HOOD"
