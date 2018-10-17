#!/bin/python

# train logistic regression classifier 

def train_classifier(X, y):
    """Train a classifier using the given training data.

    Trains a logistic regression on the input data with default parameters.
    """
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression()
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(yt, yp, pos_label=1,average='binary')
    print("  Accuracy", acc)
    print("  Precision", precision)
    print("  Recall", recall)
    print("  f1 score", f1_score)
