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
    def __init__(self, sizes):
	self.layers = len(sizes)
	self.sizes = sizes
	self._W =  [(np.random.random_sample((y, x))-0.5).astype(np.float32) for y, x in zip(sizes[:-1], sizes[1:])]
	self._b = [np.zeros((1,x)).astype(np.float32) for  x in sizes[1:]]
	
    def forward(self,X):
	rsizes = self.sizes
	rsizes.reverse()
	outputs =  [np.zeros((1,x)).astype(np.float32) for  x in rsizes[:-1]]
	i=0
	j=self.layers-2;
	while i < self.layers-2:
        	net = np.matmul( X, self._W[i] ) + self._b[i]
        	y = sigmoid(net)
		outputs[j] = y
		X=y
		i=i+1
		j=j-1
	net = np.matmul(X,self._W[i]) + self._b[i]
	z = softmax(net)
	outputs[j] = z
        return outputs

    def backpropagation(self, X, target, outputs , eta):
        d = (outputs[0] - target)
	self._W[self.layers-2] -= eta*np.matmul(outputs[1].T,d)
	i=self.layers-3;
	j=1
	while i >0:
		d = outputs[j]*(1.0-outputs[j])*np.matmul(d, self._W[i+1].T)
		self._W[i] -= eta* np.matmul(outputs[j+1].T,d)
		self._b[i] -= eta*d
		i=i-1
		j=j+1
	d = outputs[j]*(1.0-outputs[j])*np.matmul(d,self._W[i+1].T)
	self._W[i] -= eta*np.matmul(X.reshape(-1,1),d)	
	self._b[i] -= eta*d


    def predict(self,X):
	rsizes = self.sizes
        rsizes.reverse()
        outputs =  [np.zeros((1,x)).astype(np.float32) for  x in rsizes[:-1]]
        i=0
        j=self.layers-2;
        while i < self.layers-2:
                net = np.matmul( X, self._W[i] ) + self._b[i]
                y = sigmoid(net)
                outputs[j] = y
                X=y
                i=i+1
                j=j-1
        net = np.matmul(X,self._W[i]) + self._b[i]
        z = softmax(net)
        outputs[j] = z
        return outputs[0]


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

def train_and_cross_validate(sizes, num_hidden=8, n_epochs=50000,eta=0.01):
	
	X_train, X_test, y_train, y_test,X,Y = prepare_data()
	get_feature_importance(X,Y)
	nn = NN(sizes)
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
	    			outputs = nn.forward(x)
            			nn.backpropagation( x, target,outputs,eta)
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
	train_and_cross_validate([9,5,4,6]);
