import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

class Trainer:
	def init_dataset(self, file):
		tmpset = []
		with open(file, "r") as data_file:
			csv_reader = csv.reader(data_file, delimiter=",")
			for line in csv_reader:
				try:
					tmpset.append([int(line[0]), int(line[1])])
				except ValueError:
					pass
		self.dataset = np.array(tmpset, dtype = np.uint32)
		X = self.dataset[:, 0]
		Y = self.dataset[:, 1]
		return X.T, Y.T
	
	def get_matrice(self):
		X, Y = make_regression(n_samples=80, n_features=1, noise=25)
		Y = Y + abs(Y / 1.5)
#		X, Y = self.init_dataset(self, "data.csv")
		X = X.reshape(X.shape[0], 1)
		self.X = np.hstack((X, np.ones(X.shape)))
		self.X = np.hstack((X**2, self.X))
		self.Y = Y.reshape(Y.shape[0], 1)
		self.theta = np.zeros((3, 1))
#		print(self.theta)
#		print(self.theta.shape)
#		print(self.Y.shape)
#		print(self.Y)
#		print(self.X.shape)
#		print(self.X)
	
	def model(X, theta):
		tmp = X.dot(theta)
		return tmp

	def cost(self, theta):
		m = len(self.Y)
		return 1/(2*m) * np.sum((self.model(self.X, theta) - self.Y)**2)
	
	def grad(self, X, Y, theta):
		m = len(Y)
		return 1/m * X.T.dot(self.model(X, theta) - Y)
	
	def grad_descent(self, ratio, iteration):
		theta = self.theta
		self.history = np.zeros(iteration)
		for i in range(0, iteration):
			theta = theta - (ratio * self.grad(self, self.X, self.Y, theta))
			self.history[i] = self.cost(self, theta)
		self.theta = theta
		self.iter = iteration

	def show(self):
		plt.figure(figsize=(12, 8))
		plt.subplot(2, 2, 1)
		plt.scatter(self.X[:, 1], self.Y)
		#...............................
		plt.subplot(2, 2, 2)
		plt.hist2d(self.X[:, 1], self.Y.reshape(80, ), cmap='Blues')
		plt.colorbar()
		#...............................
		plt.subplot(2, 2, 3)
		plt.plot(range(self.iter), self.history)
#		plt.title('COST')
#		plt.xlabel('Iterations', c='red')
#		plt.ylabel('Error', c='red')
		#...............................
		plt.subplot(2, 2, 4)
		plt.scatter(self.X[:, 1], self.Y, label='data')
		plt.scatter(self.X[:, 1], self.model(self.X, self.theta), c='red', label='regression')
		plt.legend()
#		plt.title('DATA')
#		plt.xlabel('Km', c='red')
#		plt.ylabel('Prix', c='red')
		#...............................
		plt.show()

# ----------------------------------------------------------------------

if __name__ == "__main__":
	train = Trainer
	train.init_dataset(train, "data.csv")
	train.get_matrice(train)
	train.grad_descent(train, ratio=0.01, iteration=600)
	train.show(train)
