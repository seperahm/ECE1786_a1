import torch
import torchtext

# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)    # embedding size = 50

def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]:                         # take the top n
        print(glove.itos[idx], "\t%5.2f" % difference)

def print_nationality_adj(nation):
    print('for nation \"', nation, '\" we have the closest words:')    
    print_closest_words(glove[nation] + glove["citizen"])

nation_list = ['switzerland', 'iran', 'england', 'france', 'canada', 'russia', 'india', 'japan', 'korea', 'america', 'brazil']
for nation in nation_list:
    print_nationality_adj(nation)

