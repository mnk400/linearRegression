from numpy import genfromtxt
import numpy as np
import time
from time import sleep

class EuclideanGradient(object):
    '''
    Class to solve a linear regression problem using 
    minizing the euclidean distance between the datapoints and the hyperplane
    '''

    def __init__(self, x_without_ones, y, verbose = False, tolerance = 0.00000001):
        '''
        Constructor
        '''
        self.verbose = verbose
        self.DATAPOINTS = len(y)
        self.x_without_ones = x_without_ones
        self.y = y
        self.tolerance = tolerance
        self.initDistribution()
        self.y = np.array([self.y])
        self.y = self.y.T
        self.x = np.concatenate([self.x, self.y], 1)
        self.w = np.random.uniform(low=0.0,high=4.0,size=(len(self.x[0]) - 1,1))
        b = np.array([-1])
        self.w = np.vstack([self.w,b])

        if self.verbose:
            print("Minimizing the Euclidean Distance")

    def initDistribution(self):
        '''
        Function to append a column of ones to our input and initializing the distribution.
        '''
        self.ones = np.ones((self.DATAPOINTS,1))
        self.x = np.concatenate([self.ones, self.x_without_ones], 1)
        

    def run(self):
        '''
        Function to run the algorithm for minimizing the euclidean distances between
        the datapoints and the hyperlane.
        '''
        start = time.time()
        i = 0
        w = self.w
        k = 99.99
        while k > self.tolerance: 

            i = i + 1

            if self.verbose:
                print("Iteration " + str(i),end="\r")

            t_0 = w.T.dot(w)
            t_1 = self.x.T.dot(self.x).dot(w)
            gradient = (2/t_0) * (t_1) - (2/t_0**2) * np.transpose(w).dot(t_1) * (w)
            w_new = w - (1/self.DATAPOINTS) * 0.3 * gradient
            w_new[-1] = -1
            l = len(self.x[0]) - 1 
            y_plt = np.dot(self.x[:,0:l],w_new[:-1]) 
            k = np.dot(np.transpose(w - w_new),(w - w_new))
            w = w_new

        if self.verbose:
            print("Tolerance Reached")
            timetaken = round(time.time() - start, 3)
            print("Ran for " + str(timetaken) + " seconds" + " in " + str(i) + " iterations.")
        return w_new, y_plt

