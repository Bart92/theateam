import numpy as np
import re
import itertools
from collections import Counter

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_neutral_data():
    import pandas as pd
    neutral_examples = pd.read_csv("./totalCompanyList.txt",delimiter="\t",header=None).iloc[:,1]
    print(neutral_examples[0])
    neutral_examples = [s.strip() for s in neutral_examples]
    neutral_tweets = [clean_str(sent) for sent in neutral_examples]
    neutral_tweets = [s.split(" ") for s in neutral_tweets]
    return neutral_tweets

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = [] # PREALLOCATE FOR SPEED
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def pad_tweets(annotated, neutral, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max([max(len(x) for x in annotated), max(len(x) for x in neutral)])
    print('Padding length determined as ' +str(sequence_length))
    
    def pad(sentences, sequence_length):
        padded_sentences = [] # PREALLOCATE FOR SPEED
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences
    
    annotated_padded = pad(annotated,sequence_length)
    neutral_padded = pad(neutral,sequence_length)
    return [annotated_padded, neutral_padded]

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def build_neutral_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    z = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return z


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    neutral_tweets = load_neutral_data()
    print('Step 1: Neutral loading done')
    annotated_tweets, labels = load_data_and_labels()
    print('Step 2: Annotated loading done')
    #sentences_padded = pad_sentences(sentences)
    [annotated_padded, neutral_padded] = pad_tweets(annotated_tweets, neutral_tweets)
    print('Step 3: Padding done with '+str(len(annotated_padded))+' annotated tweets and '+str(len(neutral_padded))+' neutral tweets.')
    vocabulary, vocabulary_inv = build_vocab(annotated_padded + neutral_padded)
    print('3 done')
    x, y = build_input_data(annotated_padded, labels, vocabulary)
    z = build_neutral_data(neutral_padded, vocabulary)
    print('4 done')
    return [x, y, vocabulary, vocabulary_inv, z]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
