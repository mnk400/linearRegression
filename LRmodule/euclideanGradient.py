from numpy import genfromtxt
import numpy as np
import time
from time import sleep

class EuclideanGradient(object):
    """
    Class to solve a linear regression problem using 
    minizing the euclidean distance between the datapoints and the hyperplane
    """

    def __init__(self, dataset, target, verbose = False, tolerance = 0.00000001):
        """
        Constructor
        """

        # Setting if verbose mode on or not
        self.verbose = verbose

        # Setting the size of the target input
        self.DATAPOINTS = len(target)

        # Setting the dataset and target
        self.x_without_ones = dataset
        self.y = target

        # Setting the tolerance, default if not explicitely specified
        self.tolerance = tolerance

        # Appending a column of ones to our dataset.
        # This helps us perform calculations directly in matrix form.
        self.ones = np.ones((self.DATAPOINTS,1))
        self.x = np.concatenate([self.ones, self.x_without_ones], 1)

        # Converting target to numpy arroy and transposing
        self.y = np.array([self.y])
        self.y = self.y.T
        
        # Concatenating the targets and the dataset because 
        # when calculating the euclidean distance, we need
        # all the variables in an equation
        self.x = np.concatenate([self.x, self.y], 1)

        # generating random weights initially using a uniform distribution
        self.w = np.random.uniform(low=0.0,high=4.0,size=(len(self.x[0]) - 1,1))
        
        # setting the last weight as zero because our equations are in the form
        # ax1 + ax2 + .... - y =0
        b = np.array([-1])
        self.w = np.vstack([self.w,b])

        if self.verbose:
            print("Minimizing the Euclidean Distance")
        

    def run(self):
        """
        Function to run the algorithm for minimizing the euclidean distances between
        the datapoints and the hyperlane.
        """

        # Initializing required variables for the algorithm
        start = time.time()
        i = 0
        w = self.w
        k = 99.99

        #Running the algorithm till tolerance is reached
        while k > self.tolerance: 

            i = i + 1

            if self.verbose:
                print("Iteration " + str(i),end="\r")

            # Following is a gradient descent algorithm
            # You can find the update step at the following
            # link, https://raw.githubusercontent.com/mnk400/linearRegression/master/img/EUC.png
            t_0 = w.T.dot(w)
            t_1 = self.x.T.dot(self.x).dot(w)
            gradient = (2/t_0) * (t_1) - (2/t_0**2) * np.transpose(w).dot(t_1) * (w)
            w_new = w - (1/self.DATAPOINTS) * 0.3 * gradient
            w_new[-1] = -1
            l = len(self.x[0]) - 1 
            y_plt = np.dot(self.x[:,0:l],w_new[:-1]) 

            #tolerance is checked against the root mean square of change in weights
            k = np.dot(np.transpose(w - w_new),(w - w_new))

            #Updating the weights 
            w = w_new

        if self.verbose:
            print("Tolerance Reached")
            timetaken = round(time.time() - start, 3)
            print("Ran for " + str(timetaken) + " seconds" + " in " + str(i) + " iterations.")

        return w_new, y_plt

