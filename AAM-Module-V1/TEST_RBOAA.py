import numpy as np

from RBOAA_class import _RBOAA
from OAA_class import _OAA
from AAordinalSampler import OrdinalSampler
from OrdinalAA import RB_OAA, Ordinal_AA

from eval_measures import NMI, MCC
from synthetic_data_class import _synthetic_data

a_param = 1
b_param = 1
lr = 0.01
N = 100
M = 10
K = 4
alphaFalse = [1,2,3,4,5]
p = len(alphaFalse)
sigma = -4.6
sigma_c = np.log(1+np.exp(sigma))
bias = True
#%%
# def OrdinalSampler(N,M,alphaFalse,sigma,K,achetypeParm=1,betaParm=1,bias=True):
X, _, Z_true, A_true, beta_true = OrdinalSampler(N=N, M=M, K=K, alphaFalse=alphaFalse, achetypeParm=a_param, betaParm=b_param, sigma=sigma_c, bias=bias)

print("Computing for bachelor børge")
summery,result_RB,summery_StandardModel,result_StandardModel = RB_OAA(data=X, K=K, epokes=1000, learning_rata=lr)
# result saves the matrices needed
Z_BB_RB = result_RB["A"]
A_BB_RB = result_RB["S"]

summery,result_O = Ordinal_AA(data=X, K=K, epokes=1000, learning_rata=lr)
Z_BB_O = result_O["A"]
A_BB_O = result_O["S"]
print("Done for bachelor børge")

print("Computing for our model")
RBOAA = _RBOAA()
OAA = _OAA()
result_RBOAA = RBOAA._compute_archetypes(X=X.T, K=K, p=p, n_iter=1000, lr=lr, mute=True, columns = ["SQ"+str(i+1) for i in range(M)], early_stopping=True, with_OAA_initialization=True)
result_OAA = OAA._compute_archetypes(X=X.T, K=K, p=p, n_iter=1000, lr=lr, mute=True, columns = ["SQ"+str(i+1) for i in range(M)], early_stopping=True)
A_RBOAA = result_RBOAA.A
Z_RBOAA = result_RBOAA.Z
A_OAA = result_OAA.A
Z_OAA = result_OAA.Z
print("Done for our model")


NMI_our_RB = NMI(np.array(A_true).T, A_RBOAA)
MCC_our_RB = MCC(np.array(Z_true).T, Z_RBOAA)

NMI_BB_RB = NMI(np.array(A_true), np.array(A_BB_RB))
MCC_BB_RB = MCC(np.array(Z_true), np.array(Z_BB_RB))


NMI_our_O = NMI(np.array(A_true).T, A_OAA)
MCC_our_O = MCC(np.array(Z_true).T, Z_OAA)
NMI_BB_O = NMI(np.array(A_BB_O), np.array(A_true))
MCC_BB_O = MCC(np.array(Z_true), np.array(Z_BB_O))

print("Bachelor Børge data, Our results RBOAA:")
print("NMI: ", NMI_our_RB)
print("MCC:", MCC_our_RB)
print("--------------------------")
print("BB reuslts")
print("NMI:", NMI_BB_RB)
print("MCC:", MCC_BB_RB)

print("Bachelor Børge data, Our results OAA:")
print("NMI: ", NMI_our_O)
print("MCC:", MCC_our_O)
print("--------------------------")
print("BB reuslts")
print("NMI:", NMI_BB_O)
print("MCC:", MCC_BB_O)

syn = _synthetic_data(N=N, M=M, K=K, p=p, sigma=sigma, rb=bias, a_param=a_param, b_param=b_param)
X = syn.X
Z_true = syn.Z
A_true = syn.A

print("Computing for bachelor børge")
summery,result_RB,summery_StandardModel,result_StandardModel = RB_OAA(data=X.T, K=K, epokes=1000, learning_rata=lr)
# result saves the matrices needed
Z_BB_RB = result_RB["A"]
A_BB_RB = result_RB["S"]

summery,result_O = Ordinal_AA(data=X.T, K=K, epokes=1000, learning_rata=lr)
Z_BB_O = result_O["A"]
A_BB_O = result_O["S"]
print("Done for bachelor børge")

print("Computing for our model")
RBOAA = _RBOAA()
OAA = _OAA()
result_RBOAA = RBOAA._compute_archetypes(X=X, K=K, p=p, n_iter=1000, lr=lr, mute=True, columns = ["SQ"+str(i+1) for i in range(M)], early_stopping=True, with_OAA_initialization=True)
result_OAA = OAA._compute_archetypes(X=X, K=K, p=p, n_iter=1000, lr=lr, mute=True, columns = ["SQ"+str(i+1) for i in range(M)], early_stopping=True)
A_RBOAA = result_RBOAA.A
Z_RBOAA = result_RBOAA.Z
A_OAA = result_OAA.A
Z_OAA = result_OAA.Z
print("Done for our model")


NMI_our_RB = NMI(np.array(A_true).T, A_RBOAA)
MCC_our_RB = MCC(np.array(Z_true).T, Z_RBOAA)

NMI_BB_RB = NMI(np.array(A_true), np.array(A_BB_RB))
MCC_BB_RB = MCC(np.array(Z_true), np.array(Z_BB_RB))


NMI_our_O = NMI(np.array(A_true).T, A_OAA)
MCC_our_O = MCC(np.array(Z_true).T, Z_OAA)
NMI_BB_O = NMI(np.array(A_BB_O), np.array(A_true))
MCC_BB_O = MCC(np.array(Z_true), np.array(Z_BB_O))

print("Our data, Our results RBOAA:")
print("NMI: ", NMI_our_RB)
print("MCC:", MCC_our_RB)
print("--------------------------")
print("BB reuslts")
print("NMI:", NMI_BB_RB)
print("MCC:", MCC_BB_RB)

print("Our data, Our results OAA:")
print("NMI: ", NMI_our_O)
print("MCC:", MCC_our_O)
print("--------------------------")
print("BB reuslts")
print("NMI:", NMI_BB_O)
print("MCC:", MCC_BB_O)