from collections import Counter
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
import spacy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
                if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
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

class SGNS(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.hidden_emb = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.target_emb = torch.nn.Embedding(self.vocab_size, self.embedding_size)

        default_w = 0.5 / self.embedding_size
        self.hidden_emb.weight.data.uniform_(-default_w, default_w)
        self.target_emb.weight.data.uniform_(-default_w, default_w)
        
    def forward(self, x, t):
        center_embedding = self.hidden_emb(x)      
        target_embedding = self.target_emb(t)
        
        logit = torch.bmm(target_embedding.unsqueeze(1), center_embedding.unsqueeze(2))
        logit = logit.squeeze().reshape([-1])
        
        return logit

def train_sgns(textlist, w2i, window=5, embedding_size=8):
    # Set up a model with Skip-gram with negative sampling (predict context with word)
    # textlist: a list of strings
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create Training Data 
    X, T, Y = tokenize_and_preprocess_text(textlist, w2i, window)
    X = np.array(X)
    T = np.array(T)
    Y = np.array(Y)

    # Split the training data
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=43)
    to_tensor = lambda x: torch.from_numpy(x).to(device)
    to_tensor_float = lambda x: torch.from_numpy(x).to(device).float()
    X_train, X_test, T_train, T_test = map(to_tensor, [X_train, X_test, T_train, T_test])
    Y_train, Y_test = map(to_tensor_float, [Y_train, Y_test])
    # instantiate the network & set up the optimizer

    model = SGNS(vocab_size=len(w2i.keys()), embedding_size=embedding_size)
    model = model.to(device)

    num_epochs = 30
    batch_size = 4
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss()
    

    # training loop
    centers = X_train.split(batch_size)
    targets = T_train.split(batch_size)
    labels = Y_train.split(batch_size)

    train_loss = []
    total_val_loss = []

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for center, target, label in zip(centers, targets, labels):
            center, target, label = center.to(device), target.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(x=center, t=target)
            loss = loss_function(logits, label)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_pred = model(x=X_test, t=T_test)
        val_loss = loss_function(val_pred, Y_test).item()

        epoch_loss /= len(centers)
        train_loss.append(epoch_loss)
        total_val_loss.append(val_loss)

    # plotting network loss   
    plt.plot(range(num_epochs), train_loss, label='Training Loss')
    plt.plot(range(num_epochs), total_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return model

def visualize_embedding(embedding, most_frequent_from=20, most_frequent_to=80):
    print ("Visualizing the {} to {} most frequent words".format(most_frequent_from, most_frequent_to))
    
    # since the embeddings are ordered from most frequent words to least frequent, 
    # we can easily select a sub range of the most frequent words:
    
    selected_words = embedding[most_frequent_from:most_frequent_to, :]
    
    # The function below will reduce a vector to 2 principle components
    
    pca = PCA(n_components=2)
    
    # Transform the selected embeddings to have 2 dimensions
    
    embedding = pca.fit_transform(selected_words)
    
    X = embedding[:, 0]
    Y = embedding[:, 1]
    for i, (x,y) in enumerate(embedding):
        plt.scatter(x, y, marker='o', label=i2w[i])
        plt.annotate(i2w[i], (x, y), textcoords="offset points", xytext=(5,5), ha='center')

if __name__ == '__main__':
    with open('LargerCorpus.txt', encoding='utf-8') as f:
        corpus = f.read()
    textlist = sent_tokenize(corpus)
    filtered_lemmas, w2i, i2w, vocab = prepare_texts(corpus)
    network = train_sgns(textlist, w2i, 5, 8)
    embedding = network.out_embed.weight.data
    visualize_embedding(embedding.detach().numpy(), most_frequent_from=20, most_frequent_to=80)


    