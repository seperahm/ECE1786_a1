import torch
import torchtext

# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=300)    # embedding size = 50

def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

def print_closest_cosine_words(vec, n=5):
    similarity = torch.cosine_similarity(glove.vectors, vec.unsqueeze(0), dim=1)
    lst = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True) # sort by max similarity
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

words = ['dog', 'nurse', 'computer', 'anxiety']
for word in words:
    print("for the word \"", word, "\":")
    print_closest_words(glove[word])

print("-----------------------------------------------------")
for word in words:
    print("for the word \"", word, "\":")
    print_closest_cosine_words(glove[word])
# print("for the word \"dog\":")
# print_closest_words(glove['dog'])
# print("for the word \"nurse\":")
# print_closest_words(glove['nurse'])
# print("for the word \"computer\":")
# print_closest_words(glove['computer'])
# print("for the word \"anxiety\":")
# print_closest_words(glove['anxiety'])



