
from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np

from eval_measures import NMI
from eval_measures import MCC

from OAA_class import _OAA
from RBOAA_class import _RBOAA


### Bachelor børges archetype correlation (MCC) ###
def archetype_correlation(AT1,AT2):
    """
    :param AT1: Archetype matrix 1:  K1*m matrix
    :param AT2: Archetype matrix 2: K2*m matrix
    :return: K1*K2 matrix of correlation between correlation, K1 list of max correlation
    """

    #Make sure we have numpy arrays
    AT1=np.array(AT1.T)
    AT2 = np.array(AT2.T)

    K1,m=AT1.shape
    K2,m=AT2.shape

    correlation=np.empty((K1,K2))
    for j in range(K1):
        for i in range(K2):
            correlation[j, i] = np.corrcoef(AT1[j,:], AT2[i,:])[1, 0]

    return correlation,correlation.max(0),np.mean(correlation.max(0))

#%%

rb = False
betaParm = 100
K = 4
N = 20000
M = 21
sigmas = [-1000, -4.6, -2.97, -2.25, -1.82, -1.507, -1.05 ]
# sigmas = [-1000, -4.6]
p = 5
a_param = 1

reps = 3

OAA = _OAA()
RBOAA = _RBOAA()

NMI_OAA = np.empty((len(sigmas), reps))
NMI_RBOAA = np.empty((len(sigmas), reps))
MCC_OAA = np.empty((len(sigmas), reps))
MCC_RBOAA = np.empty((len(sigmas), reps))
MCC_OAA_alpha = np.empty((len(sigmas), reps))
MCC_RBOAA_alpha = np.empty((len(sigmas), reps))

loss_f_OAA =  np.empty((len(sigmas), reps))
loss_f_RBOAA =  np.empty((len(sigmas), reps))

MCC_OAA_BB = np.empty((len(sigmas), reps))
MCC_RBOAA_BB = np.empty((len(sigmas), reps))

for k, sigma in enumerate(sigmas):
    for i in range(reps):
        print ("iteration {0} in progress".format(k+1))
        data = _synthetic_data(N=N, M=M ,K=K, p=p, sigma=sigma, rb=rb, a_param = a_param, b_param = betaParm)
        
        X = data.X
        A_true = data.A
        Z_true = data.Z
        Z_true_alpha = data.Z_alpha
        
        # OAA
        result_OAA = OAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.01, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True, with_CAA_initialization = False)
        A_OAA = result_OAA.A
        Z_OAA = result_OAA.Z
        
        # RBOAA
        result_RBOAA = RBOAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.005, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True)
        A_RBOAA = result_RBOAA.A
        Z_RBOAA = result_RBOAA.Z
        
        # NMI
        NMI_OAA[k,i] = (NMI(A_true, A_OAA))
        NMI_RBOAA[k,i] = (NMI(A_true, A_RBOAA))
        
        # with Z true in the ordinal domain
        MCC_OAA[k,i] =(MCC(Z_true, Z_OAA))
        MCC_RBOAA[k,i] =(MCC(Z_true, Z_RBOAA))
        
        # with Z true in the alpha domain
        MCC_OAA_alpha[k,i] =(MCC(Z_true_alpha, Z_OAA))
        MCC_RBOAA_alpha[k,i] =(MCC(Z_true_alpha, Z_RBOAA))
        
        loss_f_OAA[k,i] =(result_OAA.loss[-1])
        loss_f_RBOAA[k,i] =(result_RBOAA.loss[-1])
        
        print("BACHELOR BØRGE MCC")
        MCC_OAA_BB[k,i] = (archetype_correlation(Z_true, Z_OAA)[2])
        MCC_RBOAA_BB[k,i] = (archetype_correlation(Z_true, Z_RBOAA)[2])
    
    





#%%
import matplotlib.pyplot as plt
def softplus(s):
    return np.log(1+np.exp(s))

sigmas_softplussed = [softplus(s) for s in sigmas]


# print(MCC_OAA_BB)
# print("MCC for RBOAA:")
# print(MCC_RBOAA_BB)




plt.scatter(sigmas_softplussed, np.mean(NMI_OAA,axis=1), label = "OAA")
plt.plot(sigmas_softplussed, np.mean(NMI_OAA,axis=1), '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('NMI', fontsize=18)


plt.scatter(sigmas_softplussed, np.mean(NMI_RBOAA,axis=1), label = "RBOAA")
plt.plot(sigmas_softplussed, np.mean(NMI_RBOAA,axis=1), '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('NMI', fontsize=18)
plt.title('NMI plot for the two methods no RB', fontsize = 22)
plt.legend()
plt.show()



plt.scatter(sigmas_softplussed, np.mean(MCC_OAA,axis=1), label = "OAA")
plt.plot(sigmas_softplussed, np.mean(MCC_OAA,axis=1), '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)


plt.scatter(sigmas_softplussed, np.mean(MCC_RBOAA,axis=1), label = "RBOAA")
plt.plot(sigmas_softplussed, np.mean(MCC_RBOAA,axis=1), '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)
plt.title('MCC ordinal and alpha domain no RB', fontsize = 22)

plt.legend()
plt.show()



plt.scatter(sigmas_softplussed, np.mean(MCC_OAA_alpha,axis=1), label = "OAA")
plt.plot(sigmas_softplussed, np.mean(MCC_OAA_alpha,axis=1), '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)


plt.scatter(sigmas_softplussed, np.mean(MCC_RBOAA_alpha,axis=1), label = "RBOAA")
plt.plot(sigmas_softplussed, np.mean(MCC_RBOAA_alpha,axis=1), '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)
plt.title('MCC alpha domain no RB', fontsize = 22)

plt.legend()
plt.show()




plt.scatter(sigmas_softplussed, np.mean(loss_f_OAA, axis=1), label ="OAA")
plt.plot(sigmas_softplussed, np.mean(loss_f_OAA, axis=1))
plt.xlabel('sigma', fontsize=18)
plt.ylabel('loss', fontsize=18)
plt.scatter(sigmas_softplussed,  np.mean(loss_f_RBOAA, axis=1), label = "RBOAA")
plt.plot(sigmas_softplussed,  np.mean(loss_f_RBOAA, axis=1))
plt.xlabel('sigma', fontsize=18)
plt.ylabel('loss', fontsize=18)

plt.legend()
plt.title('Loss for increasing noise no RB', fontsize = 22)



