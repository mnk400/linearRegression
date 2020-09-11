class r2(object):
    
    def calculate(self,y_p,y):
        #Method to calculate r2cal
        y_mean = y.mean()
        sst = 0
        ssreg = 0
        for i in range(len(y)):
            sst = sst + (y[i] - y_mean)**2

        for i in range(len(y)):
            ssreg = ssreg + (y_p[i] - y[i])**2

        rsquared = 1 - (ssreg/sst)
        return rsquared