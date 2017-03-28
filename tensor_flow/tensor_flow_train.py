from __future__ import print_function

# Import MNIST data

import tensorflow as tf

import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

import random

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer


learning_rate = 0.001
#learning_rate = 0.001
training_epochs = 10000
batch_size = 56
display_step = 1


# Network Parameters
n_hidden_1 = 75 # 1st layer number of features
n_hidden_2 = 50 # 2nd layer number of features
n_hidden_3 = 25
n_input = 93
n_classes = 9 


def prepare_data():
        X = []
        Y = []
        df = pd.read_csv("train.csv")
        mapping = pd.get_dummies(df['target'])
        df.drop('id', axis=1, inplace=True)
        df.drop('target', axis=1, inplace=True)
        df = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        i=0
        while i < len(df['feat_1']):
                inp = []
                out = []
                j=1
                while j <= 9:
                        out.append(mapping[j][i])
                        j=j+1
                j=1
                while j <= 93:
                        inp.append(df["feat_"+ str(j)][i])
                        j=j+1
                X.append(inp);
		Y.append(out);
                i=i+1
        g = np.random.permutation(len(X))
        X_final = []
        Y_final = []
        for p in g:
                X_final.append(X[p])
                Y_final.append(Y[p])
        X_final = np.array(X_final)
        Y_final = np.array(Y_final)
        return X_final,Y_final


def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.tanh(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.sigmoid(layer_3)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    #out_layer = tf.nn.softmax(out_layer)
    return out_layer


import matplotlib.pyplot as plt
import numpy as np

def next_batch(batch_size,i,X_train, y_train):
	return X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]

X,Y = prepare_data()
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
sss.get_n_splits(X, Y)
j=0
print("Got data");
k=0
for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        j=j+1


	
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])


	weights = {
    		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	biases = {
    		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	pred = multilayer_perceptron(x, weights, biases)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
    		sess.run(init)
		cost_values = []	
		iters = []
    		for epoch in range(training_epochs):
        		avg_cost = 0.
        		total_batch = int(len(X_train)/batch_size)
        		for i in range(total_batch):
            			batch_x, batch_y = next_batch(batch_size,i,X_train, y_train)
           			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            			avg_cost += c / total_batch
        		if epoch % display_step == 0:
            			print("Epoch:", '%04d' % (epoch+1), "cost=", \
                			"{:.9f}".format(avg_cost))
				cost_values.append(avg_cost)
				iters.append(epoch)
	        fig = plt.figure(k)
        	plt.plot(np.array(iters),np.array(cost_values))
        	plt.title("training cost")
        	plt.ylabel('cost')
        	plt.xlabel('epoch')
        	fig.savefig("otto_tensorflow_accuracy_"+str(j))
		k=k+1
    		print("Optimization Finished!")
		save_path = saver.save(sess, str(j)+"_model_x.ckpt")

    		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    		print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
