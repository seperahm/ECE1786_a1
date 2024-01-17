import torch
import torchtext

# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)    # embedding size = 50

def print_closest_cosine_words(vec, n=5):
    similarity = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0), dim=1)
    lst = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True) # sort by max similarity
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

print_closest_cosine_words(glove["cat"], n=10)