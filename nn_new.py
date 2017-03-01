import numpy as np
import pandas as pd

import random

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer


def sigmoid(X):
    return 1.0 / ( 1.0 + np.exp(-X) )

def softmax(X):
    _sum = np.exp(X).sum()
    return np.exp(X) / _sum

class NN(object):
    def __init__(self, num_input, num_hidden,num_hidden1, num_output):
        self._W1 = (np.random.random_sample((num_input, num_hidden)) - 0.5).astype(np.float32)
        self._b1 = np.zeros((1, num_hidden)).astype(np.float32)
	self._W2 = (np.random.random_sample((num_hidden,num_hidden1))- 0.5).astype(np.float32)
	self._b2 = np.zeros((1,num_hidden1)).astype(np.float32)
        self._W3 = (np.random.random_sample((num_hidden1, num_output)) - 0.5).astype(np.float32)
        self._b3 = np.zeros((1, num_output)).astype(np.float32)
	

    def forward(self,X):
        net1 = np.matmul( X, self._W1 ) + self._b1
        y = sigmoid(net1)
        net2 = np.matmul( y, self._W2 ) + self._b2
        z1 = sigmoid(net2)
	net3 = np.matmul(z1,self._W3) + self._b3
	z2 = softmax(net3)
        return z2,z1,y

    def backpropagation(self, X, target, z2, z1, y, eta):
        d3 = (z2 - target)
        d2 = z1*(1.0-z1) * np.matmul(d3, self._W3.T)
	d1 = y*(1.0 - y) * np.matmul(d2, self._W2.T)
	self._W3 -= eta * np.matmul(z1.T,d3)
        self._W2 -= eta * np.matmul(y.T,d2)
        self._W1 -= eta * np.matmul(X.reshape((-1,1)),d1)
	self._b3 -= eta*d3
        self._b2 -= eta * d2
        self._b1 -= eta * d1


    def predict(self,X):
	reg_lambda = 0.01
	net1 = np.matmul( X, self._W1 ) + self._b1
        y = sigmoid(net1)
        net2 = np.matmul( y, self._W2 ) + self._b2
        z1 = sigmoid(net2)
        net3 = np.matmul(z1,self._W3) + self._b3
        z2 = softmax(net3)
        return z2

    def calculate_loss(self,X,y):
	probs = []
	Y = []
	i=0
	data_loss =0
	for x in X:
		try:
	  		prob = self.predict(x)[0]
			probs.append(prob)
			Y.append(y[i])
		
			data_loss += np.sum(np.square(np.array(y[i])-prob))
		except:
			import ipdb; ipdb.set_trace()
		i=i+1
        return data_loss



import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt



def prepare_data():
        f = open("glass.csv",'r')
        X = []
        Y = []
        df = pd.read_csv("glass.csv")
        mapping = pd.get_dummies(df['class'])
        df.drop('class', axis=1, inplace=True)
        df = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        i=0
	elements = [ 'RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
        while i < len(df['RI']):
                inp = []
                out = []
                out.append(mapping[1][i])
                out.append(mapping[2][i])
                out.append(mapping[3][i])
		out.append(mapping[5][i])
		out.append(mapping[6][i])
		out.append(mapping[7][i])
		j=0
		while j < len(elements):
                	inp.append(df[elements[j]][i])
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
        X_train, X_test, y_train, y_test = train_test_split(X_final, Y_final, train_size=0.7)
	return (X_train, X_test, y_train,y_test,X_final,Y_final)



import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


def get_feature_importance(X,Y):
	X = np.array(X)
	Y = np.array(Y)

	forest = ExtraTreesClassifier(n_estimators=198,
                              random_state=0)

	forest.fit(X, Y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
 	            axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
    		print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],
       		color="r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), indices)
	plt.xlim([-1, X.shape[1]])
	plt.savefig("feature_imp.png");



from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

def train_and_cross_validate(n_input, n_hidden, n_hidden1, n_output, num_hidden=8, n_epochs=10000,eta=0.01):
	
	X_train, X_test, y_train, y_test,X,Y = prepare_data()
	get_feature_importance(X,Y)
	nn = NN(n_input,n_hidden,n_hidden1, n_output)
	sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

	sss.get_n_splits(X, Y)

	j=0
	for train_index, test_index in sss.split(X, Y):
    		X_train, X_test = X[train_index], X[test_index]
    		y_train, y_test = Y[train_index], Y[test_index]
    		X_train = np.array(X_train, dtype=np.float32)
    		y_train = np.array(y_train, dtype=np.float32)
    		error =[]
    		iterations = []
    		for epoch in range(n_epochs):
        		for x, target in zip(X_train,y_train):
	    			z2,z1,y = nn.forward(x)
            			nn.backpropagation( x, target,z2,z1,y,eta)
            			if epoch%100==0:
	        			error.append(nn.calculate_loss(X_test,y_test))	
	        			iterations.append(epoch)
    		plt.clf()
    		fig = plt.figure()
    		plt.plot(np.array(iterations),np.array(error))
    		plt.xlabel('epoch', fontsize=16)
    		plt.ylabel('error', fontsize=16)
    		fig.savefig('training' + '_' + str(j)+ '.png')
    

    		total=0
    		cnt=0
    		for data, data2 in zip(X_test,y_test):
        		probs = nn.forward( np.array(data, dtype=np.float32))[0]
        		total = total + 1
        		if(np.argmax(probs)==np.argmax(data2)):	
             			cnt=cnt+1	

    		accuracy = (1.0*cnt*100.0)/total
    		print("Accuracy: " + str(accuracy))
    		j=j+1

if __name__=='__main__':
	train_and_cross_validate(9,5,4,6);
