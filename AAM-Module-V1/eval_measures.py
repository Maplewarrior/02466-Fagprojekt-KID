import numpy as np

# Matrix correlation coefficient function
def MCC( A1, A2):
        _, K1 = A1.shape 
        _, K2 = A2.shape
        corr = np.zeros((K1,K2))
        for i in range(K1):
            for j in range(K2):
                corr[i][j] = np.corrcoef(A1[:,i], A2[:,j])[0][1]
        
        # max_list = []
        # for _ in range(min(K1,K2)):
        #     row, column = np.unravel_index(corr.argmax(), corr.shape)
        #     max_list.append(corr[row][column])
        #     corr = np.delete(corr, row, axis=0)
        #     corr = np.delete(corr, column, axis=1)
        
        # return np.mean(max_list)
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


# AA_type in ["OAA", "RBOAA"]
# Boundary difference measure
def BDM(b_true, b_est, AA_type):
    
    N, J = b_true.shape
    if AA_type == "OAA":
        b_est = np.array([b_est[:] for _ in range(N)])
    
    return (np.sum(np.abs(b_true-b_est)))/(N*J)

#%%


