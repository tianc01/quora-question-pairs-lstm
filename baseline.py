# baseline.py, baseline algorithm for duplicate question pairs identification
# use sum of sentence embeddings

import numpy as np
import pandas as pd
from pdb import set_trace as st
from nltk.tokenize import TweetTokenizer

import glove_embedding
import sent_embedder
import logisticRegressionClassify

def tokenPairs(qSeries1, qSeries2):
    # Return two lists
    tknzr = TweetTokenizer()
    q1tokenized = []
    q2tokenized = []
    for sent1, sent2 in zip(qSeries1, qSeries2):
        q1tokenized.append(tknzr.tokenize(sent1))
        q2tokenized.append(tknzr.tokenize(sent2))
    return q1tokenized, q2tokenized

def df_splitter(df, seed = 0, train_prop = 0.7, dev_prop = 0.15):
    """Splits the dataframe to three dataframes."""
    df_index = df.index.tolist()
    np.random.seed(seed)
    np.random.shuffle(df_index)
    train_size = int(len(df)*train_prop)
    dev_size = int(len(df)*dev_prop)
    train_index = df_index[0:train_size]
    dev_index = df_index[train_size:train_size+dev_size]
    test_index = df_index[train_size+dev_size:]
    train = df.loc[train_index]
    dev = df.loc[dev_index]
    test = df.loc[test_index]
    return train, dev, test

def get_x_y(data):
    # Tokenize sentences
    # Data type: train['question1'] Series
    # Data type: q1tokenized: list
    # print "Tokenizing training data..."
    q1tokenized, q2tokenized = tokenPairs(data['question1'], data['question2'])

    # Word Embedding
    # glove_embedding.word2vec: last input is a list of tokenized sentences
    # word2vec returns a list of numpy arrays (len(list) = num of sentences)
    # Each Numpy Array Dimension: num of words in this sentence * 50 (embedding dimension)
    # print "Word embedding..."
    W, vocab, ivocab = glove_embedding.generate()
    q1word_embedding = glove_embedding.word2vec(W, vocab, ivocab, q1tokenized)
    q2word_embedding = glove_embedding.word2vec(W, vocab, ivocab, q2tokenized)

    # sentence embedding (baseline: sum of word embeddings)
    # print "Sentence embedding..."
    sentence_embedder = sent_embedder.SumOfWordsSentEmbedder()
    # sentence_embedder = sent_embedder.LSTMSentEmbedder()

    # Obtain sentence embedding
    # Return a list of 50-dim sentence embedding (len(list) = num of sentences)
    q1sent_embedding = sentence_embedder.embed(q1word_embedding)
    q2sent_embedding = sentence_embedder.embed(q2word_embedding)

    # Concatenate two lists:
    x = np.array([np.append(x1,x2) for x1, x2 in zip(q1sent_embedding, q2sent_embedding)])
    y = np.array(data['is_duplicate'])
    return x, y

if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_csv('data/questions.csv').dropna()
    # print data.columns

    num_slice = 8000
    data = data.loc[:num_slice, :]
    print(("data.shape: {}").format(data.shape))

    # Split to train, dev and test pandas dataframe
    print("Splitting data...")
    train, dev, test = df_splitter(data)

    # Training
    print("Training classifier")
    train_x, train_y = get_x_y(train)

    clf = logisticRegressionClassify.train_classifier(train_x, train_y)

    print("Evaluating training set")
    logisticRegressionClassify.evaluate(train_x, train_y, clf)

    print("Evaluating dev set")
    dev_x, dev_y = get_x_y(dev)
    logisticRegressionClassify.evaluate(dev_x, dev_y, clf)

    print("Evaluating test set")
    test_x, test_y = get_x_y(dev)
    logisticRegressionClassify.evaluate(test_x, test_y, clf)
