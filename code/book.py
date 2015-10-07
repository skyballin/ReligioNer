import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
import re
import cPickle as pickle 

class Book():
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath, delimiter = "|", 
            skiprows=0, names = ['Book', 'Chapter', 'Verse', 'Original Text', 'Formatted Text'])
        self.unique_books = np.unique(self.df['Book'])
        self.all_text = ''
        self.parser()
        self.alltext()

    def parser(self):
        books = self.df['Book']
        indices = np.where(map(lambda x: x in self.unique_books, books))
        self.df = self.df.ix[indices]
        self.df = self.df.reset_index()
        self.df = self.df.drop('index', axis = 1)
        print self.name, "parsing beginning"
        lmtzr = WordNetLemmatizer()
        for i, text in enumerate(self.df['Original Text']):
            clean = self.clean_string(text.lower())
            clean_lemma = lmtzr.lemmatize(clean)
            self.df['Formatted Text'][i] = clean_lemma
            if i%100 == 0:
                print i
        print self.name, "parsing complete"

    def clean_string(self, input_str):
        input_str = "".join(input_str)
        clean = re.sub("[^a-zA-Z]", " ", input_str) #remove punctuation & numbers
        return clean

    def alltext(self):
        for text in self.df['Formatted Text']:
            self.all_text += text

if __name__ == "__main__":
    gita = Book('Gita', '../data/Gita/bhagavadgita.txt')
    with open('../data/gita.pkl', 'wb') as f:
        pickle.dump(gita, f)
    print "gita pickle complete"

    koran = Book('Koran', '../data/Koran/koran_text.txt')
    with open('../data/koran.pkl', 'wb') as f:
        pickle.dump(koran, f)
    print "koran pickle complete"

    torah = Book('Torah', '../data/Old Testament/oldtestament.txt')
    with open('../data/torah.pkl', 'wb') as f:
        pickle.dump(torah, f)
    print "gita pickle complete"

    newtestament = Book('New Testament', '../data/New Testament/newtestament.txt')
    with open('../data/newtestament.pkl', 'wb') as f:
        pickle.dump(newtestament, f)
    print "newtestament pickle complete"

    sutras = Book('Buddhist Sutras', '../data/Buddhism/sutras.txt')
    with open('../data/sutras.pkl', 'wb') as f:
        pickle.dump(sutras, f)
    print "sutras pickle complete"

    mormon = Book('Book Of Mormon', '../data/Mormon/BookOfMormon.txt')
    with open('../data/mormon.pkl', 'wb') as f:
        pickle.dump(mormon, f)
    print "mormon pickle complete"