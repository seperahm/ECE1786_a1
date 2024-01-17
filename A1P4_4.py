from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import spacy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm  # For progress bars

def prepare_texts(text, min_frequency=3):
    
    # Get a callable object from spacy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")
    
    # Some text cleaning. Do it by sentence, and eliminate punctuation.
    lemmas = []
    for sent in sent_tokenize(text):  # sent_tokenize separates the sentences 
        for tok in nlp(sent):         # nlp processes as in Part III
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)

    
    # Count the frequency of each lemmatized word
    freqs = Counter()  # word -> occurrence
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  # List of (word, occurrence)
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
    
    # per Mikolov, don't use the infrequent words, as there isn't much to learn in that case
    
    frequent_vocab = list(filter(lambda item: item[1]>=min_frequency, vocab))
    # print(frequent_vocab)
    # Create the dictionaries to go from word to index or vice-verse
    
    w2i = {w[0]:i for i,w in enumerate(frequent_vocab)}
    i2w = {i:w[0] for i,w in enumerate(frequent_vocab)}
    
    # Create an Out Of Vocabulary (oov) token as well
    w2i["<oov>"] = len(frequent_vocab)
    i2w[len(frequent_vocab)] = "<oov>"
    
    # Set all of the words not included in vocabulary nuas oov
    filtered_lemmas = []
    for lem in lemmas:
        if lem not in w2i:
            filtered_lemmas.append("<oov>")
        else:
            filtered_lemmas.append(lem)
    
    return filtered_lemmas, w2i, i2w, vocab


def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []
    max_word_dis = (window - 1)//2
    nlp = spacy.load("en_core_web_sm")
    dis = np.arange(window) - max_word_dis # create word range from window size
    dis = np.delete(dis, np.where(dis == 0)) # avoid self from pairing with self
    max_vocab = 2559
    for text in tqdm(textlist):
        lemma = []
        for sent in sent_tokenize(text): 
            for tok in nlp(sent):       
                if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=\'":
                    lemma.append(tok.lemma_)
                    
        lemma = [w2i[i] if i in w2i else max_vocab for i in lemma] # transfer to indices
        for i, w in enumerate(lemma):
            for d in dis:
                if i+d < len(lemma) and i+d >= 0:
                    # positive sampling
                    X.append(w)
                    T.append(lemma[i + d])
                    Y.append(1)
                    # negative sampling
                    X.append(w)
                    T.append(np.random.randint(0, max_vocab+1))
                    Y.append(0)
    return X, T, Y


if __name__ == "__main__":
    with open('LargerCorpus.txt') as f:
        corpus = f.read()
    window = 5
    lemmas, w2i, i2w, vocab = prepare_texts(corpus)
    textlist = sent_tokenize(corpus)
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, window)
    print(len(X))