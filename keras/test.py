from keras.models import Sequential
from keras.layers import Dense
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

import matplotlib.pyplot as plt

X,Y = prepare_data()
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
sss.get_n_splits(X, Y)
j=0
for train_index, test_index in sss.split(X, Y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]
	X_train = np.array(X_train, dtype=np.float32)
	y_train = np.array(y_train, dtype=np.float32)
	j=j+1

# create model
	model = Sequential()
	model.add(Dense(93, input_dim=93, init='uniform', activation='sigmoid'))
	model.add(Dense(40, init='uniform', activation='tanh'))
	model.add(Dense(30, init='uniform', activation='sigmoid'))
	model.add(Dense(9, init='uniform', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	history = model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=600, batch_size=56)
	# evaluate the model
	scores = model.evaluate(X, Y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	plt.clf()
	fig = plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	fig.savefig("otto_accuracy_"+str(j))
	# summarize history for loss
	plt.clf()
	fig = plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	fig.savefig("otto_loss_"+str(j))

