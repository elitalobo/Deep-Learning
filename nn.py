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

    def reinitalize(self,W,B):
	self._W = np.loadtxt("nn_network_weights.txt")
	self._b = np.loadtxt("nn_network_biases.txt")
	
    def forward(self,X):
	rsizes = self.sizes
	rsizes = rsizes[::-1]
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
        rsizes = rsizes[::-1]
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

    def calculate_accuracy(self,X,y):
        i=0
	total =0.0
        prediction_accuracy =0.0
	correctly_predicted = 0.0
        for x in X:
                try:
                        probs = self.predict(x)[0]
                        total +=1

                        correctly_predicted += np.argmax(probs)==np.argmax(y[i])
                except:
                        import ipdb; ipdb.set_trace()
                i=i+1
        return (correctly_predicted*1.0)/total;

	
    def calculate_confusion_matrix(self,X,Y):
	num_classes = self.sizes[-1]
	self.C = [np.zeros((1,self.sizes[-1])).astype(np.float32) for  x in range(0,self.sizes[-1])]
	for (x,y) in zip(X,Y):
		predicted = np.argmax(self.predict(x)[0]);
		actual = np.argmax(y)
		self.C[actual][0][predicted]+=1.0
	print("Confusion matrix\n")
	print(str(self.C))
	print("\n")
	for x in range(num_classes):
		try:
			if np.sum(self.C, axis=0)[0][x] != 0.0:			
				precision = (1.0*self.C[x][0][x])/np.sum(self.C,axis=0)[0][x]
				print("Precision for class " + str(x) + " : " + str(precision)+ "\n");
			if np.sum(self.C, axis=1)[0][x] != 0.0:
				recall = (1.0*self.C[x][0][x])/np.sum(self.C,axis=1)[0][x]
				print("Recall for class " + str(x) + " : " + str(recall) + "\n");
		except:
			import ipdb; ipdb.set_trace()	
			print("error")
	
	
	



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

def plot_error_and_accuracy(error,iterations, accuracies, iteration):
	plt.clf()
        fig = plt.figure()
        plt.plot(np.array(iterations),np.array(error))
        plt.xlabel('epoch')
        plt.ylabel('error')
        fig.savefig('training_error' + '_' + str(iteration)+ '.png')
        plt.clf()
        fig = plt.figure()
        plt.plot(np.array(iterations), np.array(error))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        fig.savefig('training_accuracy' + '_' + str(iteration)+ '.png')

def train(nn, X_train, y_train, X_test, y_test, iteration, num_hidden=8, n_epochs=10000, eta=0.01):
	error =[]
        iterations = []
        accuracies = []

	for epoch in range(n_epochs):
		for x, target in zip(X_train,y_train):
                	outputs = nn.forward(x)
                	nn.backpropagation( x, target,outputs,eta)
			if epoch%1000==0:
                        	error.append(nn.calculate_loss(X_test,y_test))
                        	iterations.append(epoch)
                        	accuracies.append(nn.calculate_accuracy(X_test,y_test))
                        	nn.calculate_confusion_matrix(X_test, y_test)
	plot_error_and_accuracy(error, iterations,accuracies,iteration)
	

def test(nn,X_test, y_test):
	total=0
        cnt=0
        for data, data2 in zip(X_test,y_test):
        	probs = nn.forward( np.array(data, dtype=np.float32))[0]
        	total = total + 1
        	if(np.argmax(probs)==np.argmax(data2)):
                	cnt=cnt+1

        accuracy = (1.0*cnt*100.0)/total;
        print("Accuracy: " + str(accuracy))
               

				

def train_and_cross_validate(sizes, num_hidden=8,n_epochs=50000, eta=0.01):
	
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
		train(nn,X_train, y_train, X_test, y_test, j);
		test(nn, X_test, y_test)
		np.savetxt("nn_network_weights.txt",nn._W)
		np.savetxt("nn_network_biases.txt",nn._b)
		j=j+1

if __name__=='__main__':
	train_and_cross_validate([9,5,4,6]);
