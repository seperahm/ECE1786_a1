from collections import Counter
import numpy as np
import torch
import spacy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_texts(text):    
    # Get a callable object from spaCy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")
    
    # lemmatize the text, get part of speech, and remove spaces and punctuation
    
    lemmas = [tok.lemma_ for tok in nlp(text) if tok.pos_ not in ["SPACE"]]
    
    # count the number of occurences of each word in the vocabulary
    
    freqs = Counter() 
    for w in lemmas:
        freqs[w] += 1
        
    vocab = list(freqs.items())  # List of (word, occurrence)
    
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
    # Create word->index dictionary and index->word dictionary
    
    v2i = {v[0]:i for i,v in enumerate(vocab)}
    i2v = {i:v[0] for i,v in enumerate(vocab)}
    
    return lemmas, v2i, i2v

def create_lists_from_words(word_list):
    result = []
    current_list = []

    for word in word_list:
        current_list.append(word)

        # Check if the word ends with a period (dot)
        if word.endswith('.'):
            result.append(current_list)
            current_list = []

    # Add any remaining words to the last list
    if current_list:
        result.append(current_list)

    return result

def tokenize_and_preprocess_text(textlist, v2i, window=3):
    training_set = [] #for visuals
    X = []
    Y = []
    max_word_dis = (window - 1)//2
    dis = np.arange(window) - max_word_dis # create word range from window size
    dis = np.delete(dis, np.where(dis == 0)) # avoid self from pairing with self

    for text in tqdm(create_lists_from_words(textlist)):
        for i,lemma in enumerate(text):
            for d in dis:
                if (i+d>=0 and i+d<len(lemmas) and lemma != '.' and lemmas[i+d] != '.'):
                    X.append(v2i[lemma])
                    Y.append(v2i[lemmas[i+d]])
                    training_set.append((lemma, textlist[i+d]))
    # print(len(training_set))
    return X,Y

class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # initialize word vectors to random numbers 
        
        #TO DO
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        
        # prediction function takes embedding as input, and predicts which word in vocabulary as output
        
        self.linear = torch.nn.Linear(embedding_size, vocab_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        init_weight = 0.5 / embedding_size
        self.embeddings.weight.data.uniform_(-init_weight, init_weight)  # (-1, 1)
        #TO DO
        
    def forward(self, x):
        """
        x: torch.tensor of shape (bsz), bsz is the batch size
        """
        # Compute word embeddings
        e = self.embeddings(x)   # shape: (bsz, embedding_size)
        
        # Predict context word probabilities
        logits = self.linear(e)  # shape: (bsz, vocab_size)
        
        # Apply log softmax to get probabilities
        
        #TO DO
        return logits, e

import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def train_word2vec(textlist, window, v2i, embedding_size):
    learning_rate=0.001
    batch_size=4
    num_epochs=50

    # Create the training data
    X, y = tokenize_and_preprocess_text(textlist, v2i,window=window)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    print("train_test: ", len(X_train), len(X_test), len(y_train), len(y_test))

    # instantiate the network & set up the optimizer
    model = Word2vecModel(vocab_size=len(v2i.keys()), embedding_size=embedding_size)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    # training loop
    # batches = torch.from_numpy(X_train).split(batch_size)
    # targets = torch.from_numpy(y_train).split(batch_size)
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)
    # Split the tensors into batches
    batches = []
    targets = []

    for i in range(0, len(X_train_tensor), batch_size):
        batch = X_train_tensor[i:i+batch_size]
        target = y_train_tensor[i:i+batch_size]
        batches.append(batch)
        targets.append(target)

    training_loss = []
    running_val_loss = []

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for center, context in zip(batches, targets):
            center = center.to(device)
            context = context.to(device)

            optimizer.zero_grad()
            logits, e = model(x=center) # forward
            loss = loss_function(logits, context)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_pred, _ = model(x=X_test_tensor)
        val_loss = loss_function(val_pred, y_test_tensor).item()

        epoch_loss /= len(batches)
        training_loss.append(epoch_loss)
        running_val_loss.append(val_loss)

    plt.plot(range(num_epochs), training_loss, label='Training Loss')
    plt.plot(range(num_epochs), running_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model



if __name__ == '__main__':
    with open("SmallSimpleCorpus.txt") as f:
        corpus = f.read()
    lemmas,v2i,i2v = prepare_texts(corpus)
    # print(lemmas[:50])
    network = train_word2vec(lemmas, 5, v2i, 2)
    embedding = network.embeddings.weight.data

    def visualize_embedding(embedding, most_frequent_from=0, most_frequent_to=40):
        # assert embedding.shape[1] == 2, "This only supports visualizing 2-d embeddings!"
        X = embedding[:, 0]
        Y = embedding[:, 1]
        for i, (x,y) in enumerate(embedding):
            plt.scatter(x, y, marker='o', label=i2v[i])
            plt.annotate(i2v[i], xy:(x, y), textcoords="offset points", xytext=(5,5), ha='center')
    
    visualize_embedding(embedding)

