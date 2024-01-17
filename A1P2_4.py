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
    # print("1st method for the word \"", word,
    #     "\" in the category \"",cat_string,"\" is: %.2f" %avg_of_cosine )
    # print("2nd method for the word \"", word,
    #     "\" in the category \"",cat_string,"\" is: %.2f" %cosine_of_avg)

def calculate_new_word_vector(word):
    return torch.tensor([compare_words_to_category(word, cat['colour'])[0],
            compare_words_to_category(word, cat['temperature'])[0]])
print(calculate_new_word_vector('sun'))
softmax_func = torch.softmax(calculate_new_word_vector('sun'), dim = -1)
print(softmax_func)

list_of_words2 = ['sky', 'sun', 'sea', 'ocean', 'car', 'bus', 'airplane', 'oven', 'stove','couch', 'pillow', 'ice', 'water'] 
list_of_words3 = ['sun', 'moon', 'winter', 'glow','cow','prefix','heated','wrist','ghost','cool','rain','wind']

import matplotlib.pyplot as plt
# Dictionary of words and their 2D coordinates
word_coordinates = {word : torch.softmax(calculate_new_word_vector(word), dim = -1) for word in list_of_words3}
# Create a scatter plot for each word
for word, (x, y) in word_coordinates.items():
    plt.scatter(x, y, marker='o', label=word)

    # Add the word as text beside the point
    plt.annotate(word, (x, y), textcoords="offset points", xytext=(5,5), ha='center')

# Set axis labels and a legend
plt.xlabel("Colour")
plt.ylabel("Temperature")
# plt.legend()

# Show the plot
plt.title("Word Plot")
plt.grid(True)
plt.show()
