import numpy as np
from operator import itemgetter

class SVC:
    def fit(self, X, y, selections=None):
        a = np.zeros(X.shape[0])
        ay = 0
        ayx = np.zeros(X.shape[1])
        yx = y.reshape(-1, 1)*X
        indices = np.arange(X.shape[0])
        while True:
            ydf = y*(1 - np.dot(yx, ayx.T))
            iydf = np.c_[indices, ydf]
            i = int(min(iydf[(y < 0) | (a > 0)], key=itemgetter(1))[0])
            j = int(max(iydf[(y > 0) | (a > 0)], key=itemgetter(1))[0])
            #print("i, j: {},{}".format(i, j))
            #print("yfdi, ydfj:{},{}".format(ydf[i],  ydf[j]))
            if ydf[i] >= ydf[j]:
                print('finish 1')
                break
            ay2 = ay - y[i]*a[i] - y[j]*a[j]
            #print("neway2:{}".format(ay2))
            ayx2 = ayx - y[i]*a[i]*X[i, :] - y[j]*a[j]*X[j, :]
            #print("newayx2:{}".format(ayx2))
            ai = ((1-y[i]*y[j] + y[i]*np.dot(X[i,:] - X[j, :], X[j, :]*ay2 - ayx2)) / ((X[i] - X[j])**2).sum()) 
            if ai < 0:
                ai = 0
            aj = (-ai * y[i] - ay2) * y[j]
            if aj < 0:
                aj = 0
                ai = (-aj*y[j] - ay2)*y[i]
            ay += y[i] * (ai - a[i]) + y[j]*(aj - a[j])
            #print("neway:{}".format(ay))
            ayx += y[i] * (ai - a[i]) * X[i, :] + y[j] * (aj - a[j])*X[j, :]
            #print("newayx:{}".format(ayx))
            #print("ai:{}".format(ai))
            #print("aj:{}".format(aj))
            if ai == a[i]:
                print('finish 2')
                break
            a[i] = ai
            a[j] = aj
            # print(a)
            # print("-----------------")
        self.a_ = a
        ind = a != 0.
        self.w_ = ((a[ind] * y[ind]).reshape(-1, 1) * X[ind, :]).sum(axis=0)
        self.w0_ = (y[ind] - np.dot(X[ind, :], self.w_)).sum() / ind.sum()
        print(self.w_)
        print(self.w0_)
        
    def predict(self, X):
        return np.sign(self.w0_ + np.dot(X, self.w_))
        