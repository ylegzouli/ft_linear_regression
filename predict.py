import csv
import matplotlib.pyplot as plt
import numpy as np
from trainer import Trainer

DATA_FILE = "data.csv"

def	get_price(train, X, thetas):
	thetas[1] = thetas[1] * 10000
	cost = train.model(X, thetas)
	if (cost < 0):
		cost = 0
	print("esimate price: ", cost) 

def price(train, thetas):
	command = input("km: ")
	if command == "EXIT":
		raise RuntimeError
	try:
		km = float(command)
		if (km < 0):
			km = 0
		X = np.array((km, 1))
		get_price(train, X, thetas)
	except ValueError:
		print("Error input")

def	trainer(train):
	train.get_matrice(train, DATA_FILE)
	train.grad_descent(train, ratio=0.01, iteration=3500)
	return train.get_thetas(train)

if __name__ == "__main__":
	train = Trainer
	thetas = trainer(train)
	price(train, thetas)
	thetas[1] = thetas[1] / 10000
#	coef = train.coef_det(train.get_Y(train), train.model(thetas)
#	print("coeff: ", coef)
	train.show(train)
