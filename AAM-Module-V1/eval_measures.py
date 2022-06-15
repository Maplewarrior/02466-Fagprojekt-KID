import numpy as np

# Matrix correlation coefficient function
def MCC( A1, A2):
        _, K1 = A1.shape 
        _, K2 = A2.shape
        corr = np.zeros((K1,K2))
        for i in range(K1):
            for j in range(K2):
                corr[i][j] = np.corrcoef(A1[:,i], A2[:,j])[0][1]
        
        return np.mean(corr.max(1))

def calcMI(A1, A2):
    P = A1@A2.T
    PXY = P/sum(sum(P))
    PXPY = np.outer(np.sum(PXY,axis=1), np.sum(PXY,axis=0))
    ind = np.where(PXY>0)
    MI = sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
    return MI



# Normalized mutual information function
def NMI(A1, A2):
    #krav at værdierne i række summer til 1 ???
    NMI = (2*calcMI(A1,A2)) / (calcMI(A1,A1) + calcMI(A2,A2))
    return NMI



# Boundary Measure
def BDM(b_true, b_est, AA_type):

    if b_true.ndim > 1:
        N, J = b_true.shape
        if AA_type == "OAA":
            b_est = np.array([b_est[:] for _ in range(N)])
    else:
        if AA_type == "OAA":
            J = b_true.shape[0]
            N = 1
        else:
            J = b_true.shape[0]
            N = np.shape(b_est)[0]
            b_true = np.array([b_true[:] for _ in range(N)])
    
    return (np.sum((b_true-b_est)**2))/(N*J)


# Squared Mean Distance
def SMD(A,B):
    if A.ndim > 1:
        x,y = A.shape
    else:
        x = A.shape
        y = 1
    return np.sum(abs(A-B)**2)/(x*y)

# Mean Squared Mean Distance
def MSMD(list):
    MSMD = 0
    for i in range(len(list)):
        for j in range(len(list)):
            MSMD += SMD(list[i],list[j])
    return MSMD/(len(list)**2)