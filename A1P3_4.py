from collections import Counter
import numpy as np
import torch
import spacy
from sklearn.model_selection import train_test_split

class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # initialize word vectors to random numbers 
        
        #TO DO
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        
        # prediction function takes embedding as input, and predicts which word in vocabulary as output
        
        self.linear = torch.nn.Linear(embedding_size, vocab_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        init_weight = 0.5 / self.embedding_size
        self.embedding.weight.data.uniform_(-init_weight, init_weight)  # (-1, 1)
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
        log_probs = self.log_softmax(logits)
        
        #TO DO
        return logits, e

vocab_size = 11
embedding_size = 2
vec_model = Word2vecModel(vocab_size, embedding_size)