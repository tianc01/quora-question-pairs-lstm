## Improving Semantic Question Matching of Quora Dataset by LSTM Encoders

The task is to identify duplicate questions for a [Quora question pair dataset](https://www.kaggle.com/c/quora-question-pairs).A neural network model
with LSTM sentence embedding is implemented, which is proposed by Bowman [(Bowman et al., 2015)](https://nlp.stanford.edu/pubs/snli_paper.pdf) and are developed on a large
corpus: Stanford Natural Language Inference (SNLI). The method is evaluated on a subset of Quora’s dataset.

### Files
`baseline.py`: Sum of Word Embedding + Logistic Regression Classifier

`nn_sum_embedding.py`: Sum of Word Embedding + Neural Network Classifier

`nn_lstm_embedding.py`: LSTM Embedding + Neural Network Classifier

`preprocess.py`: tokenization, GloVe word embedding, padding

`glove_embedding.py`: Implement GloVe word embedding

`sent_embedder.py`: Define Class Sentence Embedder

`logisticRegressionClassify.py`: Logistic Regression classifier

