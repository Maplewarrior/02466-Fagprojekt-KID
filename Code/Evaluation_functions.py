import numpy as np


def archetype_correlation(AT1,AT2):
    """
    :param AT1: Archetype matrix 1:  K1*m matrix
    :param AT2: Archetype matrix 2: K2*m matrix
    :return: K1*K2 matrix of correlation between correlation, K1 list of max correlation
    """

    #Make sure we have numpy arrays
    AT1=np.array(AT1)
    AT2 = np.array(AT2)

    K1,m=AT1.shape
    K2,m=AT2.shape

    correlation=np.empty((K1,K2))
    for j in range(K1):
        for i in range(K2):
            correlation[j, i] = np.corrcoef(AT1[j,:], AT2[i,:])[1, 0]

    return correlation,correlation.max(0),np.mean(correlation.max(0))

def NMI(S1,S2):
    """
    :param S1: reconstuction matrix 1: N*K1 matrix
    :param S2: reconstuction matrix 2: N*K2 matrix
    :return: Normalised mutial information
    """

    #Make sure we have numpy arrays
    S1=np.array(S1)
    S2 = np.array(S2)

    N, K = S1.shape
    NMI=I(S1, S2, N, K)*2/(I(S1, S1, N, K)+I(S2, S2, N, K))


    return NMI

def I(S1, S2, N, K):
    Pkk = S1.T @ S2 / N
    I=0

    Pd1=np.sum(S1,axis=0)/np.sum(S1,axis=(0,1))
    Pd2=np.sum(S2,axis=0)/np.sum(S2,axis=(0,1))

    for k1 in range(K):
        for k2 in range(K):
            I+=Pkk[k1,k2]*np.log(Pkk[k1,k2]/(Pd1[k1]*Pd2[k2]))


    return I


def ResponsBiasCompereson(beta1, beta2):
    """

    :param beta1: list or matrix of beta values.
    :param beta2: list or matrix of beta values.
    :return: matrix of list of difference between values, mean if difference
    :Note if beta1 is a list beta 2 must also be a list, but if beta1 is a matrix and beta 2 is a list the operation will
    be row-vise on beta1
    """
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)

    if len(beta1.shape) == 1:
        beta1 = beta1[1:-1]
    else:
        beta1 = beta1[:, 1:-1]

    if len(beta2.shape) == 1:
        beta2 = beta2[1:-1]
    else:
        beta2 = beta2[:, 1:-1]

    dif = np.abs(beta1 - beta2)
    return dif, np.mean(dif)


def Ordinal_reconstuction(X1,X2,beta1=None,beta2=None):
    """
    :param X1: N*M matrix
    :param X2: N*M matrix
    :param beta1: If X1 is not ordinal data arrange in 0 to J provided beta values to convert to ordinal data
    :param beta2: If X2 is not ordinal data arrange in 0 to J provided beta values to convert to ordinal data
    :return: value for between 1 and 0 for each point reconstructed correctly
    """

    if beta1 is not None:
        X1=np.sum(np.array([beta1[j] < X1 for j in range(1, len(beta1))]),0)

    if beta2 is not None:
        X2 = np.sum(np.array([beta2[j] < X2 for j in range(1, len(beta2))]), 0)

    return np.mean(X1==X2)

if __name__ == '__main__':
    import json
    import os
    M2simsavedir = r"C:\Users\Andre\OneDrive - Danmarks Tekniske Universitet\Bachelor project\Experiment2"

    M2simfiles = ["sigma1true", "sigma1sample0", "sigma1sample1", "sigma1sample2", "sigma1sample3", "sigma1sample4"]
    M2simData = []
    for file in M2simfiles:
        with open(os.path.join(M2simsavedir, file), "r") as file:
            M2simData.append(json.load(file))

    S1 = M2simData[0]['S_True']
    S2 = M2simData[1]['S']
    NMI(S1, S2)