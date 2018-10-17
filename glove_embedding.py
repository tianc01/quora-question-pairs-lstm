# 50 dimmension embedding
import argparse
import numpy as np

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='data/vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='data/vectors.txt', type=str)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in list(vectors.items()):
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)

def word2vec(W, vocab, ivocab, sent_list):
    # Return a list of numpy arrays, where every numpy array is a 
    # matrix of word embeddings for a sepecific sentence
    wordembedding_sent = [None] * len(sent_list)
    for i, sent in enumerate(sent_list):
        newsent = [term.lower() for term in sent if term.lower() in vocab]
        vec_array = np.zeros((len(newsent),W.shape[1]))
        for idx, term in enumerate(newsent):
            vec_array[idx] = W[vocab[term], :]
        wordembedding_sent[i] = vec_array
    return wordembedding_sent