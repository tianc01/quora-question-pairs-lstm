import numpy as np
from pdb import set_trace as st

class SentEmbedder:
    def embed(self, words_embedding):
        pass

class SumOfWordsSentEmbedder(SentEmbedder):
    def __init__(self):
        pass

    def embed(self, words_embedding):
        sent_embedding = np.empty([len(words_embedding),words_embedding[0].shape[1]])
        for i,sent in enumerate(words_embedding):
            sent_embedding[i] = np.sum(sent,axis=0)
        return sent_embedding

    

