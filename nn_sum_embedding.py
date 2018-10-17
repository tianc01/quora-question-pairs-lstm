import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.metrics
from pdb import set_trace as st

import glove_embedding
import sent_embedder


def tokenPairs(qSeries1, qSeries2):
    # Return two lists
    from nltk.tokenize import TweetTokenizer
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
    print("Tokenizing training data...")
    q1tokenized, q2tokenized = tokenPairs(data['question1'], data['question2'])

    # Word Embedding
    # glove_embedding.word2vec: last input is a list of tokenized sentences
    # word2vec returns a list of numpy arrays (len(list) = num of sentences)
    # Each Numpy Array Dimension: num of words in this sentence * 50 (embedding dimension)
    print("Word embedding...")
    W, vocab, ivocab = glove_embedding.generate()
    q1word_embedding = glove_embedding.word2vec(W, vocab, ivocab, q1tokenized)
    q2word_embedding = glove_embedding.word2vec(W, vocab, ivocab, q2tokenized)

    # sentence embedding (baseline: sum of word embeddings)
    print("Sentence embedding...")
    sentence_embedder = sent_embedder.SumOfWordsSentEmbedder()
    # sentence_embedder = sent_embedder.LSTMSentEmbedder()

    # Obtain sentence embedding
    # Return a list of 100-dim sentence embedding (len(list) = num of sentences)
    q1sent_embedding = sentence_embedder.embed(q1word_embedding)
    q2sent_embedding = sentence_embedder.embed(q2word_embedding)

    # Concatenate two lists:
    x = np.array([np.append(x1,x2) for x1, x2 in zip(q1sent_embedding, q2sent_embedding)])
    y = np.array(data['is_duplicate'])
    return x, y

def multilayer_perceptron(X, weights, biases):
    # Hidden layer with tanh activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.layers.batch_normalization(layer_1)
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden layer with tanh activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.layers.batch_normalization(layer_2)
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    # Hidden layer with tanh activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.layers.batch_normalization(layer_3)
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

def prepare_data():
    print("Loading data...")
    data = pd.read_csv('data/questions.csv').dropna()

    num_slice = 400
    data = data.loc[:num_slice-1, :]
    print("data.shape: ", data.shape)

    # Split to train, dev and test pandas dataframe
    print("Splitting data...")
    train, dev, test = df_splitter(data)

    # Training
    print("Transform data...")
    train_X, train_label = get_x_y(train)
    dev_X, dev_label = get_x_y(dev)
    test_X, test_label = get_x_y(test)

    n_labels = 2
    train_y = np.zeros((train_label.shape[0], n_labels))
    dev_y = np.zeros((dev_label.shape[0], n_labels))
    test_y = np.zeros((test_label.shape[0], n_labels))
    train_y[np.arange(train_label.shape[0]), train_label] = 1
    dev_y[np.arange(dev_label.shape[0]), dev_label] = 1
    test_y[np.arange(test_label.shape[0]), test_label] = 1

    return train_X, train_y, dev_X, dev_y, test_X, test_y

if __name__ == "__main__":

    train_X, train_y, dev_X, dev_y, test_X, test_y = prepare_data()
    # Hyper-parameters we will not tune
    dim_input = train_X.shape[1]
    dim_output = 2
    dim_hidden_1 = 200
    dim_hidden_2 = 200
    dim_hidden_3 = 200

    # Hyper-parameters for tuning
    learning_rate = 0.001
    batch_size = 128
    epoch = 500
    beta = 0.01 # L2 regularization
    keep_prob = 0.8
    stop = 1e-7

    tf.reset_default_graph()

    # Define input and output
    X = tf.placeholder(tf.float32, [None, dim_input])
    y = tf.placeholder(tf.float32, [None, dim_output]) # one hot vector


    # Define weights and biases
    weights = {
        'h1': tf.Variable(tf.random_normal([dim_input, dim_hidden_1])),
        'h2': tf.Variable(tf.random_normal([dim_hidden_1, dim_hidden_2])),
        'h3': tf.Variable(tf.random_normal([dim_hidden_2, dim_hidden_3])),
        'out': tf.Variable(tf.random_normal([dim_hidden_3, dim_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([dim_hidden_1])),
        'b2': tf.Variable(tf.random_normal([dim_hidden_2])),
        'b3': tf.Variable(tf.random_normal([dim_hidden_3])),
        'out': tf.Variable(tf.random_normal([dim_output]))
    }

    # Prediction
    yp = multilayer_perceptron(X, weights, biases)

    # Define regularizer
    regularizer = tf.zeros([], tf.float32)
    for key in list(weights.keys()):
        regularizer += beta*tf.nn.l2_loss(weights[key])
    for key in list(biases.keys()):
        regularizer += beta*tf.nn.l2_loss(biases[key])

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yp, labels=y)+regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    # correct_prediction: tensor of logical values. 1 or 0 for each data
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(yp, 1))
    # tf.cast: change logical value to float
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Execute the graph
    # Initialization
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    num_batches = int(len(train_y)/batch_size)
    avg_cost_by_epoch = [None]*epoch
    for i in range(epoch):
        avg_cost_by_epoch[i] = 0.
        for j in range(num_batches):
            idx = 0
            batch_X = train_X[idx:idx+batch_size]
            batch_y = train_y[idx:idx+batch_size]
            idx += batch_size
            _, c = sess.run([optimizer, cost],feed_dict={X: batch_X, y: batch_y})
            # Compute average loss
            avg_cost_by_epoch[i] += c / num_batches
        print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost_by_epoch[i]))
        # print("Train Accuracy:", sess.run(accuracy,feed_dict={X1: train_X1, X2: train_X2, y: train_y}))
        # print("Dev Accuracy:", sess.run(accuracy,feed_dict={X1: dev_X1, X2: dev_X2, y: dev_y}))
        if i > 1:
            diff = abs(avg_cost_by_epoch[i]-avg_cost_by_epoch[i-1])
            if diff <= stop:
                break

    # Evaluate
    print(("Test Accuracy:", sess.run(accuracy,feed_dict={X: test_X, y: test_y})))

    #metrics
    y_p = tf.argmax(yp, 1)
    test_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X: test_X, y: test_y})
    y_true = np.argmax(test_y,1)

    print("Test accuracy:", test_accuracy)
    print("Precision", sklearn.metrics.precision_score(y_true, y_pred))
    print("Recall", sklearn.metrics.recall_score(y_true, y_pred))
    print("f1_score", sklearn.metrics.f1_score(y_true, y_pred))

