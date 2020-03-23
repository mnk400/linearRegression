from numpy import genfromtxt
import numpy as np
from time import sleep
from sklearn.datasets import make_regression as mr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class WML(object):
    '''
    Class to solve a linear regression problem using 
    calculating the maximum likelihood and solving for weights
    '''

    plotFlag = True
    tolerance = 0.00000001
    

    def __init__(self,x_without_ones,y):
        '''
        Constructor
        '''
        self.DATAPOINTS = len(y)
        self.x_without_ones = x_without_ones
        self.y = y
        self.generateDistribution()
        self.y = np.array([self.y])
        self.y = self.y.T
        self.w = np.random.uniform(low=-1.0,high=4.0,size=(len(self.x[0]),1))

    def generateDistribution(self):
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
        w_ml = np.linalg.inv(self.x.T.dot(self.x)).dot(self.x.T.dot(self.y))
        y_plt = np.dot(self.x,w_ml)
        return w_ml, y_plt

if __name__ == "__main__":
    FEATURES = 1
    DATAPOINTS = 1000
    NOISE = 13
    x, y = mr(n_samples = DATAPOINTS, n_features = FEATURES, n_informative= 1, n_targets= 1, noise = NOISE)
    obj = WML(x,y,DATAPOINTS)
    obj.run()    
