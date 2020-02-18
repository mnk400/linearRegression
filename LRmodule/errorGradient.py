from numpy import genfromtxt
import numpy as np
from time import sleep
from sklearn.datasets import make_regression as mr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class ErrorGradient(object):
    
    plotFlag = True
    tolerance = 0.00000001
    

    def __init__(self,x_without_ones,y,d):
        self.DATAPOINTS = d
        self.x_without_ones = x_without_ones
        self.y = y
        self.generateDistribution()
        self.y = np.array([self.y])
        self.y = self.y.T
        self.w = np.random.uniform(low=-1.0,high=4.0,size=(len(self.x[0]),1))

    def generateDistribution(self):
        self.ones = np.ones((self.DATAPOINTS,1))
        self.x = np.concatenate([self.ones, self.x_without_ones], 1)

    def run(self):
        i = 0
        w = self.w
        k = 99.99
        while k > self.tolerance:
            i = i + 1
            #Calculate gradient
            gradient = (np.dot(np.linalg.inv(np.dot(np.transpose(self.x),self.x)),(np.dot(np.transpose(self.x),(np.dot(self.x,w) - self.y)))))
            w_new = w - 0.3*gradient
            #Calculate predicted y values
            self.y_plt = np.dot(self.x,w_new) 
            #Calculate k
            k = np.dot(np.transpose(w - w_new),(w - w_new))
            #print(k)
            #Update weights
            w = w_new
            #print(w_new_new)
        return w_new, self.y_plt
