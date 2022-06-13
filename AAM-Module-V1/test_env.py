
from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np
import os
import matplotlib.pyplot as plt

from eval_measures import NMI
from eval_measures import MCC

from OAA_class import _OAA
from RBOAA_class import _RBOAA
from TSAA_class import _TSAA



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
"""
#### Code for testing NMI and MCC for OAA and RBOAA with various Z ####
#######################################################################


plots = _plots()
rb = False
betaParm = 100
K = 10
N = 25000
M = 21
sigmas = [-1000, -4.6, -2.97, -2.25, -1.82, -1.507, -1.05 ]
# sigmas = [-1000, -4.6]
p = 5
a_param = 1
data = _synthetic_data(N=N, M=M ,K=K, p=p, sigma=-1.507, rb=rb, a_param = a_param, b_param = betaParm)        
X = data.X
A_true = data.A
Z_true = data.Z
Z_true_alpha = data.Z_alpha


NMI_TSAA = []
MCC_TSAA_hat = []
MCC_TSAA_hat_alpha = []

MCC_TSAA = []
MCC_TSAA_alpha = []


TSAA = _TSAA()
result_TSAA = TSAA._compute_archetypes(X=X, K=K, n_iter=5000, lr=0.005, mute=False, columns=data.columns)

plots._loss_plot(loss=result_TSAA.RSS, type="TSAA")



Z_hat_TSAA = result_TSAA.X_hat @ result_TSAA.B
Z_TSAA = result_TSAA.Z
A_TSAA = result_TSAA.A

NMI_TSAA.append(NMI(A_TSAA, A_true))

MCC_TSAA.append(MCC(Z_true, Z_TSAA))
MCC_TSAA_alpha.append(MCC(Z_true_alpha, Z_TSAA))

"""
"""

#### TEST FOR b_param = high, rb = True and rb = False ####
###########################################################

bias = [True, False]
b_param = 100000
OAA = _OAA()
RBOAA = _RBOAA()

for rb in bias:
    data = _synthetic_data(N=N, M=M ,K=K, p=p, sigma=-4.6, rb=rb, a_param = a_param, b_param = betaParm)    
    X = data.X
    A_true = data.A
    Z_true = data.Z
    
    result_OAA = OAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.01, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True, with_CAA_initialization = False)
    A_OAA = result_OAA.A
    Z_OAA = result_OAA.Z
    loss_OAA = result_OAA.loss
    
    result_RBOAA = RBOAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.01, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True)
    A_RBOAA = result_RBOAA.A
    Z_RBOAA = result_RBOAA.Z
    loss_RBOAA = result_RBOAA.loss

#%%
"""


#%%
rb = True
betaParms = [10, 1000, 100000]
K = 4
N = 1000
M = 10
sigmas = [-1000, -4.6, -2.97, -2.25, -1.82, -1.507, -1.05]
# sigmas = [-1000, -4.6]
p = 5
a_params = [0.85, 1]

reps = 2

OAA = _OAA()
RBOAA = _RBOAA()



path = os.getcwd()
foldername = "plots_test_OAA_RBOAA"
path = os.path.join(path, foldername)


def softplus(s):
    return np.log(1+np.exp(s))

sigmas_softplussed = [softplus(s) for s in sigmas]
#%%


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


for betaParm in betaParms:
    for a_param in a_params:
        for k, sigma in enumerate(sigmas):
            for i in range(reps):
                
                data = _synthetic_data(N=N, M=M ,K=K, p=p, sigma=sigma, rb=rb, a_param = a_param, b_param = betaParm)
                
                X = data.X
                A_true = data.A
                Z_true = data.Z
                
                # OAA
                result_OAA = OAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.05, mute=True, columns=data.columns, with_synthetic_data = True, early_stopping = True, with_CAA_initialization = False)
                A_OAA = result_OAA.A
                Z_OAA = result_OAA.Z
                
                # RBOAA
                result_RBOAA = RBOAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.01, mute=True, columns=data.columns, with_synthetic_data = True, early_stopping = True)
                A_RBOAA = result_RBOAA.A
                Z_RBOAA = result_RBOAA.Z
                
                # NMI
                NMI_OAA[k,i] = (NMI(A_true, A_OAA))
                NMI_RBOAA[k,i] = (NMI(A_true, A_RBOAA))
                
                # with Z true in the ordinal domain
                MCC_OAA[k,i] =(MCC(Z_true, Z_OAA))
                MCC_RBOAA[k,i] =(MCC(Z_true, Z_RBOAA))
                
                
                loss_f_OAA[k,i] =(result_OAA.loss[-1])
                loss_f_RBOAA[k,i] =(result_RBOAA.loss[-1])
                

        ### NMI PLOT ###
        plt.scatter(sigmas_softplussed, np.mean(NMI_OAA,axis=1), label = "OAA")
        plt.plot(sigmas_softplussed, np.mean(NMI_OAA,axis=1), '--')
        plt.xlabel('sigma', fontsize=18)
        plt.ylabel('NMI', fontsize=18)
        
        plt.scatter(sigmas_softplussed, np.mean(NMI_RBOAA,axis=1), label = "RBOAA")
        plt.plot(sigmas_softplussed, np.mean(NMI_RBOAA,axis=1), '--')
        plt.xlabel('sigma', fontsize=18)
        plt.ylabel('NMI', fontsize=18)
        plt.title('NMI plot for OAA, RBOAA', fontsize = 22)
        plt.legend()
        file = "NMI w. a_param={0} and b_param={1}.png".format(a_param, betaParm)
        plt.savefig(os.path.join(path, file))
        # plt.show()
        
        
        ### MCC PLOT ###
        plt.scatter(sigmas_softplussed, np.mean(MCC_OAA,axis=1), label = "OAA")
        plt.plot(sigmas_softplussed, np.mean(MCC_OAA,axis=1), '--')
        plt.xlabel('sigma', fontsize=18)
        plt.ylabel('MCC', fontsize=18)
        
        
        plt.scatter(sigmas_softplussed, np.mean(MCC_RBOAA,axis=1), label = "RBOAA")
        plt.plot(sigmas_softplussed, np.mean(MCC_RBOAA,axis=1), '--')
        plt.xlabel('sigma', fontsize=18)
        plt.ylabel('MCC', fontsize=18)
        plt.title('MCC plot for OAA and RBOAA', fontsize = 22)
    
        plt.legend()
        file = "MCC w. a_param={0} and b_param={1}.png".format(a_param, betaParm)
        plt.savefig(os.path.join(path, file))
        # plt.show()
        
        
        
        ### LOSS PLOT ###
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
        file = "Loss w. a_param={0} and b_param={1}.png".format(a_param, betaParm)
        plt.savefig(os.path.join(path, file))
        # plt.show()
                
                
            
    



# print(MCC_OAA_BB)
# print("MCC for RBOAA:")
# print(MCC_RBOAA_BB)




# plt.scatter(sigmas_softplussed, np.mean(NMI_OAA,axis=1), label = "OAA")
# plt.plot(sigmas_softplussed, np.mean(NMI_OAA,axis=1), '--')
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('NMI', fontsize=18)


# plt.scatter(sigmas_softplussed, np.mean(NMI_RBOAA,axis=1), label = "RBOAA")
# plt.plot(sigmas_softplussed, np.mean(NMI_RBOAA,axis=1), '--')
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('NMI', fontsize=18)
# plt.title('NMI plot for the two methods no RB', fontsize = 22)
# plt.legend()
# plt.show()



# plt.scatter(sigmas_softplussed, np.mean(MCC_OAA,axis=1), label = "OAA")
# plt.plot(sigmas_softplussed, np.mean(MCC_OAA,axis=1), '--')
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('MCC', fontsize=18)


# plt.scatter(sigmas_softplussed, np.mean(MCC_RBOAA,axis=1), label = "RBOAA")
# plt.plot(sigmas_softplussed, np.mean(MCC_RBOAA,axis=1), '--')
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('MCC', fontsize=18)
# plt.title('MCC ordinal and alpha domain no RB', fontsize = 22)

# plt.legend()
# plt.show()



# plt.scatter(sigmas_softplussed, np.mean(MCC_OAA_alpha,axis=1), label = "OAA")
# plt.plot(sigmas_softplussed, np.mean(MCC_OAA_alpha,axis=1), '--')
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('MCC', fontsize=18)


# plt.scatter(sigmas_softplussed, np.mean(MCC_RBOAA_alpha,axis=1), label = "RBOAA")
# plt.plot(sigmas_softplussed, np.mean(MCC_RBOAA_alpha,axis=1), '--')
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('MCC', fontsize=18)
# plt.title('MCC alpha domain no RB', fontsize = 22)

# plt.legend()
# plt.show()




# plt.scatter(sigmas_softplussed, np.mean(loss_f_OAA, axis=1), label ="OAA")
# plt.plot(sigmas_softplussed, np.mean(loss_f_OAA, axis=1))
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('loss', fontsize=18)
# plt.scatter(sigmas_softplussed,  np.mean(loss_f_RBOAA, axis=1), label = "RBOAA")
# plt.plot(sigmas_softplussed,  np.mean(loss_f_RBOAA, axis=1))
# plt.xlabel('sigma', fontsize=18)
# plt.ylabel('loss', fontsize=18)

# plt.legend()
# plt.title('Loss for increasing noise no RB', fontsize = 22)


# #%%
# a_param = 2
# print("rb={0}, a_param={1}".format(rb, a_param))
