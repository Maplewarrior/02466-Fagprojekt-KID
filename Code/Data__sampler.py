

"""
Function usede for testing.
Sample ordinal data with known parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
def ordinal_sampler(n,j):

    alpha=np.sort(np.random.normal(5,2,j).clip(0))

    B=np.array([(alpha[i]+alpha[i+1])/2 for i in range(j-1)])

    Y=np.random.normal(5,2,n).clip(0)

    X=[]

    for y in Y:
        if y<B[0]:
            X.append(1)
        elif y>B[j-2]:
            X.append(j)
        else:
            for i in range(1,j-1):
                if y>B[i-1] and y<B[i]:
                    X.append(i+1)
    return X,alpha,Y

def ordinal_matrix(N,M,j,paired=True):

    if paired:
        alpha = np.sort(np.random.normal(5, 2, j).clip(0))
        B = np.array([(alpha[i] + alpha[i + 1]) / 2 for i in range(j - 1)])

    X=np.empty((N,M))
    Y=np.random.normal(5,2,(N,M)).clip(0)

    for m in range(M):
        for n in range(N):
            if Y[n,m]<B[0]:
                X[n,m]=1
            elif Y[n,m]>B[j-2]:
                X[n,m]=j
            else:
                for i in range(1,j-1):
                    if Y[n,m]>B[i-1] and Y[n,m]<B[i]:
                        X[n,m]=i+1
    return X,alpha,Y

def sample_data(n,m,ordinal=False):
    if ordinal:
        for i in range(m):
            if i==0:
                X=np.array(ordinal_sampler(n,5)[0])
            else:
                X=np.vstack((X,np.array(ordinal_sampler(n,5)[0])))
        X=X.T

    else:
        mean=np.ones(m)
        #cov=np.identity(m)
        cov=np.array([[1,0.8],[0.8,1]])
        X=np.random.multivariate_normal(mean,cov,size=n)
    return X

def clusters(n,m,k):
    mean=np.random.multivariate_normal(np.zeros(m), cov=np.identity(m)*10, size=k)

    for i in range(k):
        if i==0:
            X=np.random.multivariate_normal(mean[i], cov=np.identity(m), size=np.int(n/k))
        else:
            X=np.vstack((X,np.random.multivariate_normal(mean[i], cov=np.identity(m), size=np.int(n/k))))
    return X



if __name__ == '__main__':
    from OrdinalPlots import ordinal_plot
    data,cuts,Y=ordinal_sampler(1000,5)
    ordinal_plot(data,cuts)