from numpy import genfromtxt
import numpy as np
from time import sleep
from sklearn.datasets import make_regression as mr
import matplotlib.pyplot as plt

class EuclideanGradient(object):
    '''
    Class to solve a linear regression problem using 
    minizing the euclidean distance between the datapoints and the hyperplane
    '''
    
    DATAPOINTS = 10000
    plotFlag = True
    tolerance = 0.00000001
    k = 9999.99

    def __init__(self,x_without_ones,y,d):
        '''
        Constructor
        '''
        self.DATAPOINTS = d
        self.x_without_ones = x_without_ones
        self.y = y
        self.initDistribution()
        self.y = np.array([self.y])
        self.y = self.y.T
        self.x = np.concatenate([self.x, self.y], 1)
        self.w = np.random.uniform(low=0.0,high=4.0,size=(len(self.x[0]) - 1,1))
        b = np.array([-1])
        self.w = np.vstack([self.w,b])

    def initDistribution(self):
        '''
        Function to append a column of ones to our input and init'ing the distribution
        '''
        self.ones = np.ones((self.DATAPOINTS,1))
        self.x = np.concatenate([self.ones, self.x_without_ones], 1)

    def run(self):
        '''
        Function to run the algorithm 
        '''
        i = 0
        w = self.w
        while self.k > self.tolerance: 
            i = i + 1
            print(i)
            t_0 = w.T.dot(w)
            t_1 = self.x.T.dot(self.x).dot(w)
            gradient = (2/t_0) * (t_1) - (2/t_0**2) * np.transpose(w).dot(t_1) * (w)
            w_new = w - (1/self.DATAPOINTS) * 0.3 * gradient
            w_new[-1] = -1
            #print(w_new)
            l = len(self.x[0]) - 1 
            y_plt = np.dot(self.x[:,0:l],w_new[:-1]) 
            self.k = np.dot(np.transpose(w - w_new),(w - w_new))
            #print(self.k)
            w = w_new

        return w_new, y_plt

