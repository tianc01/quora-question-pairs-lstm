import pickle
import numpy as np
import pandas as pd
import glove_embedding

class PreprocessData():
    def __init__(self, data_size, max_len_1, max_len_2):
        self.data_size = data_size
        self.max_len_1 = max_len_1
        self.max_len_2 = max_len_2

    def tokenPairs(self, qSeries1, qSeries2):
        # Return two lists
        from nltk.tokenize import TweetTokenizer
        tknzr = TweetTokenizer()
        q1tokenized = []
        q2tokenized = []
        for sent1, sent2 in zip(qSeries1, qSeries2):
            q1tokenized.append(tknzr.tokenize(sent1))
            q2tokenized.append(tknzr.tokenize(sent2))
        return q1tokenized, q2tokenized

    def embeddingData(self):
        print("Loading data...")
        data = pd.read_csv('data/questions.csv').dropna()
        data = data.loc[:self.data_size-1, :]
        print("data.shape: ", data.shape)

        print("Tokenizing training data...")
        q1tokenized, q2tokenized = self.tokenPairs(data['question1'], data['question2'])

        print("Word embedding...")
        W, vocab, ivocab = glove_embedding.generate()
        q1word_embedding = glove_embedding.word2vec(W, vocab, ivocab, q1tokenized)
        q2word_embedding = glove_embedding.word2vec(W, vocab, ivocab, q2tokenized)

        # Transform to panda dataframe
        q1word_embedding = pd.Series(q1word_embedding)
        q2word_embedding = pd.Series(q2word_embedding)

        data_embedded = pd.DataFrame()
        data_embedded['q1word_embedding'] = q1word_embedding
        data_embedded['q2word_embedding'] = q2word_embedding
        data_embedded['is_duplicate'] = data['is_duplicate']

        return data_embedded

    def storeData(self):
        # Load data
        data_embedded = self.embeddingData()
        dim_embedded = data_embedded['q1word_embedding'].iloc[0].shape[1]

        # Check sentence lengths
        sent1_length = np.empty(self.data_size, dtype=int)
        for i,word_emb in enumerate(data_embedded['q1word_embedding']):
            sent1_length[i] = word_emb.shape[0]
        y = np.bincount(sent1_length)
        ii = np.nonzero(y)[0]
        # zip(ii,y[ii])

        sent2_length = np.empty(self.data_size, dtype=int)
        for i,word_emb in enumerate(data_embedded['q2word_embedding']):
            sent2_length[i] = word_emb.shape[0]
        y = np.bincount(sent2_length)
        ii = np.nonzero(y)[0]
        # zip(ii,y[ii]) 

        # Padding and Trimming
        print('Padding and Trimming ...')
        q1w_embed_pad = [None]*self.data_size
        for i,word_emb in enumerate(data_embedded['q1word_embedding']):
            dim_padding = int(self.max_len_1-sent1_length[i])
            if dim_padding > 0:
                q1w_embed_pad[i] = np.append(word_emb, np.zeros((dim_padding,dim_embedded)), axis = 0)
            else:
                q1w_embed_pad[i] = word_emb[:self.max_len_1]
            
        q2w_embed_pad = [None]*self.data_size
        for i,word_emb in enumerate(data_embedded['q2word_embedding']):
            dim_padding = int(self.max_len_2-sent2_length[i])
            if dim_padding > 0:
                q2w_embed_pad[i] = np.append(word_emb, np.zeros((dim_padding,dim_embedded)), axis = 0)
            else:
                q2w_embed_pad[i] = word_emb[:self.max_len_2]

        data_padded = pd.DataFrame()
        data_padded['q1w_embed_pad'] = q1w_embed_pad
        data_padded['q2w_embed_pad'] = q2w_embed_pad
        data_padded['is_duplicate'] = data_embedded['is_duplicate']

        pickle.dump(data_padded, open( "data/data_emb200_pad_%d.p" % self.data_size , "wb" ))