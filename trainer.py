import csv
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = "data.csv"

class	Trainer:
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
	
	def get_matrice(self, file):
		X, Y = self.init_dataset(self, file)
		X = X.reshape(X.shape[0], 1)
		X = X / 10000
		Y = Y / 10000
		self.X = np.hstack((X, np.ones(X.shape)))
		self.Y = Y.reshape(Y.shape[0], 1)
		self.theta = np.zeros((2, 1))
	
	def model(X, theta):
		tmp = X.dot(theta)
		return tmp

	def cost(self, theta):
		m = len(self.Y)
		return 1/(2*m) * np.sum((self.model(self.X * 10000, theta) - self.Y * 10000)**2)
	
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
		self.X = self.X * 10000
		self.Y = self.Y * 10000

#	def coef_det(y, pred):
#		u = ((y - pred)**2).sum()
#		v = ((y - y.mean())**2).sum()
#		return 1 - u/v

	def show(self):
		plt.figure(figsize=(12, 8))
		plt.subplot(2, 2, 1)
		plt.scatter(self.X[:, 0], self.Y)
		#...............................
		plt.subplot(2, 2, 2)
		plt.hist2d(self.X[:, 0], self.Y.reshape(24, ), cmap='Blues')
		plt.colorbar()
		#...............................
		plt.subplot(2, 2, 3)
		plt.plot(range(self.iter), self.history)
		#...............................
		plt.subplot(2, 2, 4)
		plt.scatter(self.X[:, 0], self.Y, label='data')
		plt.plot(self.X[:, 0], self.model(self.X, self.theta), c='red', label='regression')
		plt.legend()
		#...............................
		plt.show()

	def get_thetas(self):
		return self.theta

	def	get_Y(self):
		return self.Y

# ----------------------------------------------------------------------

if __name__ == "__main__":
	train = Trainer
	train.get_matrice(train, DATA_FILE)
	train.grad_descent(train, ratio=0.01, iteration=3500)
	train.show(train)
