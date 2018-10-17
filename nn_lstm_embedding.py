import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics
import sklearn.preprocessing
from preprocess import PreprocessData
from pdb import set_trace as st

def LSTM_embedder(X):
    # Creat LSTM RNN cell
    cell = tf.nn.rnn_cell.LSTMCell(dim_state_lstm,state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=lstm_keep_prob)
    # Store the outputs and states; we discard the states
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    # Transpose the output to switch batch size with sequence size; [0,1] to [1,0]
    rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    # Take the last output of the sequence; it accounts for all the previous outputs 
    last = tf.gather(rnn_outputs, int(rnn_outputs.get_shape()[0]) - 1)
    return last

def lower_mapping(S,weights,biases):
    # Additional tanh layer to map 200d to 100d
    lS = tf.add(tf.matmul(S, weights['lower']), biases['lower'])
    lS = tf.layers.batch_normalization(lS)
    lS = tf.nn.tanh(lS)
    lS = tf.nn.dropout(lS, keep_prob)
    return lS

# Function to concatenate the pair
def concatenate_pair(S1,S2):
    S = tf.concat([S1, S2], 1) 
    return S

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

def df_splitter(df, seed = 2010, train_prop = 0.7, dev_prop = 0.15):
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
    # Concatenate two lists:
    # x1 = data['q1word_embedding'].tolist()
    # x2 = data['q2word_embedding'].tolist()
    x1 = np.array(data['q1w_embed_pad'].tolist())
    x2 = np.array(data['q2w_embed_pad'].tolist())
    y = np.array(data['is_duplicate'].tolist())
    
    return x1,x2,y

def standardize(train_X, dev_X, test_X):
    trainX_reshaped = np.reshape(train_X, (train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
    mean_trainX = np.mean(trainX_reshaped, axis = 0)
    std_trainX = np.std(trainX_reshaped, axis = 0)
    norm_train_X = (trainX_reshaped - mean_trainX)/std_trainX
    train_X = np.reshape(norm_train_X, (train_X.shape[0], train_X.shape[1], train_X.shape[2]))

    devX_reshaped = np.reshape(dev_X, (dev_X.shape[0], dev_X.shape[1]*dev_X.shape[2]))
    testX_reshaped = np.reshape(test_X, (test_X.shape[0], test_X.shape[1]*test_X.shape[2]))
    norm_dev_X = (devX_reshaped - mean_trainX)/std_trainX
    norm_test_X = (testX_reshaped - mean_trainX)/std_trainX
    dev_X = np.reshape(norm_dev_X, (dev_X.shape[0], dev_X.shape[1], dev_X.shape[2]))
    test_X = np.reshape(norm_test_X, (test_X.shape[0], test_X.shape[1], test_X.shape[2]))
    return train_X, dev_X, test_X

def balance_model(df,seed = 0):
    data_one = df.loc[df['is_duplicate'] == 1]
    data_one_index = data_one.index.tolist()
    data_zero = df.loc[df['is_duplicate'] == 0]
    data_zero_index = data_zero.index.tolist()
    np.random.seed(seed)
    np.random.shuffle(data_zero_index)
    return_size = sum(df['is_duplicate'] == 1)
    data_zero_return_index = data_zero_index[:return_size]
    
    data_return_index = data_one_index+data_zero_return_index
    np.random.shuffle(data_return_index)
    
    data_return = df.loc[data_return_index]
    return data_return

def prepare_data(data_size, max_len_1, max_len_2, preprocess, balance_data):
    if preprocess:
        pd = PreprocessData(data_size, max_len_1, max_len_2)
        pd.storeData()

    data_padded = pickle.load(open("data/data_emb200_pad_%d.p" % data_size, "rb" ))
    print(data_padded.columns)

    if balance_data:
        data = balance_model(data_padded)
    else:
        data = data_padded
    
    # Split to train, dev and test pandas dataframe
    # print "Splitting embedded data..."
    train, dev, test = df_splitter(data)

    # Obtain train, dev, test data
    train_X1,train_X2,train_label = get_x_y(train)
    dev_X1,dev_X2,dev_label = get_x_y(dev)
    test_X1,test_X2,test_label = get_x_y(test)

    # Normalize x1, x2
    train_X1,dev_X1,test_X1 = standardize(train_X1,dev_X1,test_X1)
    train_X2,dev_X2,test_X2 = standardize(train_X2,dev_X2,test_X2)

    print('train non-duplicate prop:', 1-np.mean(train_label))
    print('dev non-duplicate prop:', 1-np.mean(dev_label))
    print('test non-duplicate prop:', 1-np.mean(test_label))

    # Transform y to one hot vector
    n_labels = 2
    train_y = np.zeros((train_label.shape[0], n_labels))
    dev_y = np.zeros((dev_label.shape[0], n_labels))
    test_y = np.zeros((test_label.shape[0], n_labels))
    train_y[np.arange(train_label.shape[0]), train_label] = 1
    dev_y[np.arange(dev_label.shape[0]), dev_label] = 1
    test_y[np.arange(test_label.shape[0]), test_label] = 1

    return data_padded, train, train_X1,train_X2,train_y,dev_X1,dev_X2,dev_y,test_X1,test_X2,test_y

if __name__ == "__main__":
    data_size = 8000
    max_len_1 = 28
    max_len_2 = 31

    data_padded, train, train_X1,train_X2,train_y,dev_X1,dev_X2,dev_y,test_X1,test_X2,test_y = \
    prepare_data(data_size, max_len_1, max_len_2, preprocess = True, balance_data = False)

    # Hyper-parameters we will not tune
    n_labels = 2
    dim_embedded = train_X1.shape[2]
    dim_output = 2
    dim_state_lstm = 32
    dim_lower = 100
    dim_input = dim_lower*2
    dim_hidden_1 = 200
    dim_hidden_2 = 200
    dim_hidden_3 = 200

    # Hyper-parameters we will tune
    learning_rate = 0.001
    stop = learning_rate*0.01 # stop = 1e-7
    keep_prob_value = 0.5
    lstm_keep_prob = 1.0
    beta = 0.01 # L2 regularization
    batch_size = 128
    epoch = 500

    tf.reset_default_graph()

    # Define input and output
    X1 = tf.placeholder(tf.float32, [None, max_len_1,  dim_embedded])
    X2 = tf.placeholder(tf.float32, [None, max_len_2,  dim_embedded])
    y = tf.placeholder(tf.float32, [None, dim_output]) # one hot vector
    keep_prob = tf.placeholder(tf.float32)

    # Define weights and biases
    weights = {
        'lower': tf.Variable(tf.random_normal([dim_state_lstm, dim_lower])),
        'h1': tf.Variable(tf.random_normal([dim_input, dim_hidden_1])),
        'h2': tf.Variable(tf.random_normal([dim_hidden_1, dim_hidden_2])),
        'h3': tf.Variable(tf.random_normal([dim_hidden_2, dim_hidden_3])),
        'out': tf.Variable(tf.random_normal([dim_hidden_3, dim_output]))
    }
    biases = {
        'lower': tf.Variable(tf.random_normal([dim_lower])),
        'b1': tf.Variable(tf.random_normal([dim_hidden_1])),
        'b2': tf.Variable(tf.random_normal([dim_hidden_2])),
        'b3': tf.Variable(tf.random_normal([dim_hidden_3])),
        'out': tf.Variable(tf.random_normal([dim_output]))
    }


    # Generate sentence embedding and map to 100d
    with tf.variable_scope('sent1'):
        S1 = LSTM_embedder(X1)
        lS1 = lower_mapping(S1,weights, biases)
    with tf.variable_scope('sent2'):
        S2 = LSTM_embedder(X2)
        lS2 = lower_mapping(S2,weights, biases)
    # Concatenat sentences
    S = concatenate_pair(lS1,lS2)

    # Prediction
    yp = multilayer_perceptron(S, weights, biases)
    # yp = tf.nn.softmax(multilayer_perceptron(S, weights, biases))

    # Define regularizer
    regularizer = tf.zeros([], tf.float32)
    for key in list(weights.keys()):
        regularizer += beta*tf.nn.l2_loss(weights[key])
    for key in list(biases.keys()):
        regularizer += beta*tf.nn.l2_loss(biases[key])
     
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yp, labels=y))+regularizer
    # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yp, labels=y)+regularizer)
    # cost = -tf.reduce_sum(y * tf.log(tf.clip_by_value(yp,1e-10,1.0)))+regularizer
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
    display_step = 10
     
    num_batches = int(len(train_y)/batch_size)
    avg_cost_by_epoch = [None]*epoch
    train_acc = [None]*epoch
    dev_acc = [None]*epoch
    test_acc = [None]*epoch

    for i in range(epoch):
        avg_cost_by_epoch[i] = 0.
        for j in range(num_batches):
            idx = 0
            batch_X1 = train_X1[idx:idx+batch_size]
            batch_X2 = train_X2[idx:idx+batch_size]
            batch_y = train_y[idx:idx+batch_size]

            idx += batch_size
            _, c = sess.run([optimizer, cost],feed_dict={X1: batch_X1, X2: batch_X2, y: batch_y, keep_prob: keep_prob_value})
            # Compute average loss
            avg_cost_by_epoch[i] += c / num_batches
        # train_acc[i] = sess.run(accuracy,feed_dict={X1: train_X1, X2: train_X2, y: train_y, keep_prob: 1.0})
        # dev_acc[i] = sess.run(accuracy,feed_dict={X1: dev_X1, X2: dev_X2, y: dev_y, keep_prob: 1.0})
        # test_acc[i] = sess.run(accuracy,feed_dict={X1: test_X1, X2: test_X2, y: test_y, keep_prob: 1.0})
        
        if i % display_step == 0:
            print("Epoch: {} cost = {}".format(i, avg_cost_by_epoch[i]))
            # print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost_by_epoch[i]), "train acc=",  "{:.9f}".format(train_acc[i]), "dev acc=",  "{:.9f}".format(dev_acc[i]), "test acc=",  "{:.9f}".format(test_acc[i]))
        if i > 1:
            diff = abs(avg_cost_by_epoch[i]-avg_cost_by_epoch[i-1])
            if diff <= stop:
                break

    #metrics
    y_p = tf.argmax(yp, 1)
    test_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X1: test_X1, X2: test_X2, y: test_y, keep_prob:1.0})
    y_true = np.argmax(test_y,1)

    print("Test Accuracy:", test_accuracy)
    print("Precision", sklearn.metrics.precision_score(y_true, y_pred))
    print("Recall", sklearn.metrics.recall_score(y_true, y_pred))
    print("f1_score", sklearn.metrics.f1_score(y_true, y_pred))