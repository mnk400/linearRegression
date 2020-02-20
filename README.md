# Least Squares and Maximum Likelihood for Linear Regression
Solving the linear regression problem using various techniques.

- ### Ordinary Least Squares 
Minimizing the square differences between values of estimated Y and observed Y. Solved using gradient descent, update rule is as following.

![](img/OLS.png)

Code accepts a dataset matrix(which can be multidimensional), a targets matrix(only supports single dimensional targets) and returns a vector of weights and a matrix of predicted Y's.
```Python
from LRmodule import errorGradient
regression = errorGradient.ErrorGradient(Dataset,Targets,[Size of the Dataset])
weights, predicted_y = regression.run()
```

- ### Minimizing Euclidean Distances | Total Least Squares 
Minimizing the square of Euclidean Distances between the data-points and the hypderplane, a form of Total Least Squares. Solved using gradient descent, update rule is as following.

![](img/EUC.png)

Code has the same properties as before.
```Python
from LRmodule import euclideanGradient
regression = euclideanGradient.EuclideanGradient(Dataset,Targets,[Size of the Dataset])
weights, predicted_y = regression.run()
```

- ### Maximum Likelihood
Forming a closed form solution by calculating maximum liklihood from the log-liklihood of the sum of squared errors. Weights are be calculated accordign to the formula below.

![](img/WML.png)

Code has the same properties as before.
```Python
from LRmodule import wML
regression = wML.WML(Dataset,Targets,[Size of the Dataset])
weights, predicted_y = regression.run()
```

- #### Calculating R-Squared
Repository also contains a module to calcuale r-squared with.
```Python
from rsquare import r2
r = r2.r2()
value = r.calculate(Predicted_Y, Original_Y)
```

main.py contains demo code which creates regression using scilearns make_regression to test the above programs with. Also compares the r-squared values and plots graphs depicting and comparing the predicted linear equations(Only if using 1 feature).