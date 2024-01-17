from collections import Counter
import numpy as np
import torch
import spacy
from sklearn.model_selection import train_test_split

# prepare text using the spacy english pipeline (see https://spacy.io/models/en)
# we'll use it to lemmatize the text, and determine which part of speech each
# lemmatize edits words to become the 'root' word - e.g. holds -> hold;  rubs->rub
# part of speech indicates if the item is a verb, nooun, punctuation, space and so on.
# make sure that the text sent to spacy doesn't end with a period immediately followed by a newline,
# instead, make sure there is a space between the period and the newline, so that the period 
# is correctly identified as punctuation.

def prepare_texts(text):    
    # Get a callable object from spaCy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")
    
    # lemmatize the text, get part of speech, and remove spaces and punctuation
    
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["PUNCT", "SPACE"]]
    
    # count the number of occurences of each word in the vocabulary
    
    freqs = Counter() 
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  # List of (word, occurrence)
    
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
#     print(vocab)
    
    # Create word->index dictionary and index->word dictionary
    
    v2i = {v[0]:i for i,v in enumerate(vocab)}
    i2v = {i:v[0] for i,v in enumerate(vocab)}
    
    return lemmas, v2i, i2v

def generate_training_set(corpus, window=3):
    training_set = []
    lemmas, v2i, i2v = prepare_texts(corpus)
    max_word_dis = (window - 1)//2
    dis = np.arange(window) - max_word_dis # create word range from window size
    dis = np.delete(dis, np.where(dis == 0)) # avoid self from pairing with self
    for i,lemma in enumerate(lemmas):
        for d in dis:
            if (i+d>=0 and i+d<len(lemmas)):
                training_set.append((lemma, lemmas[i+d]))
    return training_set

with open("SmallSimpleCorpus.txt") as f:
    corpus = f.read()

training_set = generate_training_set(corpus)
print(training_set)