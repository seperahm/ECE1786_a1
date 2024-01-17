import torch
import torchtext

# The first time you run this will download a ~823MB file
glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                              dim=50)    # embedding size = 50


def compare_words_to_category(word, meaning_cat):
    avg_of_cosine = float(sum(
        torch.cosine_similarity(glove[meaning].unsqueeze(0), glove[word].unsqueeze(0), dim=1)
        for meaning in meaning_cat) / len(meaning_cat))
    
    cosine_of_avg = float(torch.cosine_similarity(
        (sum(glove[meaning] for meaning in meaning_cat)/len(meaning_cat)).unsqueeze(0),
        glove[word].unsqueeze(0),
        dim=1))
    return avg_of_cosine, cosine_of_avg

cat = {}
cat['colour'] = ['colour','red', 'green', 'blue', 'yellow']
cat['temperature'] = ['temperature', 'heat', 'hot', 'warm', 'cool', 'cold']

word = 'sea'
cat_string = 'temperature'
list_of_words = ['frozen', 'sun', 'sea', 'airplane', 'car', 'burn', 'water', 'gold', 'oven', 'pillow'] 
for word in list_of_words:
    avg_of_cosine, cosine_of_avg = compare_words_to_category(word, cat[cat_string])
    print("1st method for the word \"", word,
        "\" in the category \"",cat_string,"\" is: %.2f" %avg_of_cosine )
    print("2nd method for the word \"", word,
        "\" in the category \"",cat_string,"\" is: %.2f" %cosine_of_avg)
    # print("the cosine similarity of the word \"", word,
    #     "\" with the average embedding of all of the words in the category \"",cat_string,"\" is: %.2f" %cosine_of_avg )

print(torch.cosine_similarity(glove['hot'].unsqueeze(0), glove['cold'].unsqueeze(0), dim=1))