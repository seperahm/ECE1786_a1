# ECE1786 Fall 2023 - Assignment 1: Word Embeddings - Properties, Meaning, and Training

## Introduction
Welcome to Assignment 1 of ECE1786 for the Fall 2023 term. This assignment focuses on the exploration and understanding of word embeddings, a fundamental concept in natural language processing and machine learning. Through this assignment, you will dive into the intricacies of word embeddings, exploring their properties, meanings, and the methods used for training them.

## Goals
The primary objectives of this assignment are:
1. **Understanding the Properties of Word Embeddings**: Gain insights into how word embeddings are structured and how their properties reflect linguistic relationships.
2. **Exploring the Meaning in Word Embeddings**: Learn how to extract and interpret the semantic meaning from these embeddings.
3. **Hands-on Training of Word Embeddings**: Implement and train word embeddings using techniques like the Skip-Gram method.
4. **Advanced Concepts in Word Embedding Training**: For those interested, delve into more advanced topics like Skip-Gram with Negative Sampling.

Through these goals, the assignment aims to provide a comprehensive understanding of word embeddings, from their theoretical underpinnings to practical implementation and training methods.

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
