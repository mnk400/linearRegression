from LRmodule import errorGradient, euclideanGradient, wML
from rsquare import r2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_regression as mr

FEATURES = 1
DATAPOINTS = 1000
NOISE = 30
ITERATIONS = 1

if __name__ == "__main__":
    i = 0
    r = r2.r2()
    print("Features: " + str(FEATURES) + " Datapoints: " + str(DATAPOINTS) + " Noise: " + str(NOISE)  + " Datasets: " + str(ITERATIONS))
    while i<ITERATIONS:
        i = i +1
        print("For Dataset: " +str(i))

        #generating regression example
        x, y = mr(n_samples = DATAPOINTS, n_features = FEATURES, n_informative= 1, n_targets= 1, noise = NOISE)
        
        print(x)
        #sleep(100)
        #Instance for error squared class
        print("Calculating errorGradient")
        error = errorGradient.ErrorGradient(x,y, verbose=False)
        #Predicting
        weights_er, predicted_y = error.run()
        
        #Instance for squared euclidean class
        print("Calculating euclideanGradient")
        euc = euclideanGradient.EuclideanGradient(x,y, verbose=True)
        #Predicting
        weights_eu, predicted_y_eu = euc.run()

        #Instance for squared euclidean class
        print("Calculating Maximum likelihood")
        ml = wML.WML(x,y)
        #Predicting
        weights_ml, predicted_y_ml = ml.run()

        print("R-Square value for Error Gradient for " + str(i) + "st run" + str(r.calculate(predicted_y,y)))
        print("R-Square value for Euclidean Gradient for " + str(i) + "st run" + str(r.calculate(predicted_y_eu,y)))
        print("R-Square value for Maximum Likelihood for " + str(i) + "st run" + str(r.calculate(predicted_y_ml,y)))
    

    if FEATURES == 1 and ITERATIONS == 1:
        fig = plt.figure() 
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
        ax1.set_xlabel("x1")
        ax2.set_xlabel("x1")
        ax3.set_xlabel("x1")
        ax1.set_ylabel("y")
        ax2.set_ylabel("y")
        ax3.set_ylabel("y")
        ax1.scatter(x, y, color = "pink")
        ax2.scatter(x, y, color = "skyblue")
        ax3.scatter(x, y, color = "darkseagreen")
        ax1.plot(x, predicted_y, color ="red")
        ax2.plot(x, predicted_y_eu, color ="slateblue")
        ax3.plot(x, predicted_y_ml, color ="green")
        red_patch = mpatches.Patch(color="red", label = "Sum of squared errors \nr-Square: " +str(r.calculate(predicted_y,y)) )
        blue_patch = mpatches.Patch(color="slateblue", label = "Sum of squared Euclidian distances\nr-Square: " +str(r.calculate(predicted_y_eu,y)))
        green_patch = mpatches.Patch(color="green", label = "Minimizing sum of squared errors\nr-Square: " +str(r.calculate(predicted_y_ml,y)))
        ax1.legend(handles = [red_patch])
        ax2.legend(handles = [blue_patch])
        ax3.legend(handles = [green_patch])
        plt.show()
    input()


