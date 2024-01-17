# ECE1786 Fall 2023 - Assignment 1: My Exploration of Word Embeddings

## Introduction
Hello! This repository contains my work for Assignment 1 of the ECE1786 course, offered in Fall 2023. In this assignment, I delved into the fascinating world of word embeddings, exploring their properties, meanings, and training methodologies. This journey through natural language processing and machine learning has been both challenging and rewarding, offering deep insights into how machines can interpret and process human language.

## Goals and Achievements
Throughout this assignment, I aimed to achieve the following goals:

1. **Deciphering the Properties of Word Embeddings**: I explored the structure and linguistic relationships encapsulated in word embeddings, understanding their foundational role in NLP.

2. **Extracting Meaning from Word Embeddings**: I learned how to interpret semantic meanings from word embeddings, which was particularly intriguing as it bridged the gap between raw data and human language comprehension.

3. **Practical Training of Word Embeddings**: I implemented and trained word embeddings using the Skip-Gram method. This hands-on experience was invaluable in understanding the nuances of embedding training.

4. **Exploring Advanced Concepts**: As an additional challenge, I explored advanced topics like Skip-Gram with Negative Sampling, which broadened my understanding of efficient training methods in NLP.

I am excited to share my solutions and insights gained from this assignment. I hope this repository serves as a useful resource for anyone interested in the intricacies of word embeddings and their applications in natural language processing.

---

Feel free to explore my code and the accompanying documentation. Feedback and discussions are always welcome!


### Assignment Report (ECE1786_Assignment_1_Report.pdf)
- **Description**: A comprehensive PDF document that provides summaries and answers for questions and the resulting models.

### A1P1_2.py
- **Description**: Implements the `print_closest_cosine_words` function. This function prints the N-most similar words using cosine similarity, contrasting it with Euclidean distance. It includes examples comparing the 10 most cosine-similar words to 'dog' and 'computer'.

### A1P1_3.py
- **Description**: Contains code for generating the second word in a word-pair relationship using word embeddings. Involves selecting a unique relationship, generating 10 examples, and commenting on the results.

### A1P2_1.py
- **Description**: Features the `compare_words_to_category` PyTorch function. This function computes the cosine similarity of a word against a set of words describing a category in two different ways.

### A1P3_4.py
- **Description**: Includes the `Word2VecModel` class code, which provides trained embeddings. The class is defined with an embedding size of 2, and the file details the total number of parameters in the model.

### A1P3_5.py
- **Description**: Contains the `train_word2vec` training loop function. This function sets up and executes the training process for a Word2Vec model, including data preparation, learning rate selection, and plotting loss curves.

### A1P4_4.py
- **Description**: Features the `tokenize_and_preprocess_text` function, creating positive and negative samples for training the Skip-Gram with Negative Sampling model. It involves generating labeled training examples from a corpus.

### A1P4_6.py
- **Description**: Includes the `SkipGramNegativeSampling` model class code. This model takes tokens of two words (word and context) as input and predicts a binary outcome.

### A1P4_7.py
- **Description**: Contains the `train_sgns` function for training the Skip-Gram with Negative Sampling model. Similar to `A1P3_5.py`, it involves the entire training process setup, including data preparation and plotting training/validation curves.
