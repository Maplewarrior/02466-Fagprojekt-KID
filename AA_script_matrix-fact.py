# Conventional AA

import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.sparse import csr_matrix

data = pd.read_csv('Data/ESS8_data.csv')



# Rescale and convert data
attributeNames = data.columns
v = 0



X = data.values
X = data.to_numpy()
X = X[:1000, :8]

X = X[:,2:]

X = X.astype(np.float64)

X = X.T





# # Number of archetypes
# nc = 6

# # A = k x N and B = n x K
# A = np.ones((nc, N))
# B = np.ones((N,nc))

# # Initialize Z
# idxs = np.random.permutation(N+1)[:nc]
# Z = X[:, idxs]


# def applySoftmax(X):
#     return softmax(X)

# def unitVec(nc, idx):
#     v = np.empty((nc))
#     v[idx] = 1
#     return v
    

# A = applySoftmax(A)
# B = applySoftmax(B)

import numpy as np
from cvxopt import solvers, base





# Authors: Christian Thurau
# License: BSD 3 Clause
"""  
MatrixFact Singular Value Decomposition.
    SVD : Class for Singular Value Decomposition
    pinv() : Compute the pseudoinverse of a Matrix
     
"""
import time
import scipy.sparse
import numpy as np



"""
Overview of dimensions:
    
- noc = number of archetypes
- M = input shape
- N = number of samples


X = M x N

A = noc x N
B = noc x N

Z = B @ X.T

"""

## Setting variables

## number of archetypes
noc = 6

M, N = X.shape

# set cvxopt options
solvers.options['show_progress'] = False


# Z = N x noc
def InitZandB(X, N, noc):
    beta = np.random.random((noc, N))
    beta /= beta.sum(axis = 0)
    Z = np.dot(beta, X).T
    Z = np.random.random((N, noc))
    
    return Z, beta

def InitA(noc, N):
    A = np.random.random((noc, N))
    A /= A.sum(axis = 0)
    
    return A





def update_A(X, Z, A, N, noc):
    
    EQb = base.matrix(1.0, (1,1))
    # float64 required for cvxopt
    HA = base.matrix(np.float64(np.dot(Z.T, Z)))
    INQa = base.matrix(-np.eye(noc))
    INQb = base.matrix(0.0, (noc,1))
    EQa = base.matrix(1.0, (1, noc))
    
    def update_single_a(A,i):
        FA = base.matrix(np.float64(np.dot(-Z.T, X[:,i])))
        al = solvers.qp(HA, FA, INQa, INQb, EQa, EQb)
        A[i,:] = np.array(al['x']).reshape((1, noc))
        # self.H[:,i] = np.array(al['x']).reshape((1, self._num_bases))  
        
    
    for i in range(N):
        A[i,:] = update_single_a(A,i)
    
    return A
        
def update_B(X, Z, A, B, N, noc):
    HB = base.matrix(np.float64(np.dot(X[:,:].T, X[:,:])))
    EQb = base.matrix(1.0, (1, 1))
    W_hat = np.dot(X, np.linalg.pinv(A))
    INQa = base.matrix(-np.eye(noc))
    INQb = base.matrix(0.0, (noc, 1))
    EQa = base.matrix(1.0, (1, noc))
    
    def update_single_b(i):
        FB = base.matrix(np.float64(np.dot(-X.T, )))
        be = solvers.qp(HB, FB, INQa, INQb, EQa, EQb)
        B[i,:] = np.array(be['x']).reshape((1, N))
        
        return B[i,:]
        
    for i in range(noc):
        B[i,:] = update_single_b(i)
    
    
    Z = np.dot(B, X.T).T
    
    return B, Z



#%%
## number of archetypes
noc = 6
N, M = X.shape



Z, B = InitZandB(X, N, noc)

A = InitA(noc, N)


for iteration in range(10):
    
    A = update_A(X, Z, A, N, noc)
    # print(A.shape)
    
    B, Z = update_B(X, Z, A, B, N, noc)
    
