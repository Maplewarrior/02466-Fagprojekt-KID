
from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np

from eval_measures import NMI
from eval_measures import MCC

from OAA_class import _OAA
from RBOAA_class import _RBOAA

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
betaParm = 1000
K = 4
N = 10000
M = 21
epokes = 1000
sigmas = [-1000, -4.6, -2.97, -2.25, -1.82, -1.507, -1.05 ]
# sigmas = [-1000, -4.6]
p = 5
a_param = 1

OAA = _OAA()
RBOAA = _RBOAA()

NMI_OAA = []
NMI_RBOAA = []
MCC_OAA = []
MCC_RBOAA = []
MCC_OAA_alpha = []
MCC_RBOAA_alpha = []

loss_f_OAA =  []
loss_f_RBOAA =  []

MCC_OAA_BB = []
MCC_RBOAA_BB = []

for k, sigma in enumerate(sigmas):
    print("We are on the", k, "iteration")
    data = _synthetic_data(N=N, M=M ,K=K, p=p, sigma=sigma, rb=rb, a_param = a_param, b_param = betaParm)
    X = data.X
    A_true = data.A
    
    Z_true = data.Z
    # print("ORDINAL ZZZZZ")
    # print(Z_true)
    Z_true_alpha = data.Z_alpha
    # print("ZZZZZ IN ALPHA DOMAIN")
    # print(Z_true_alpha)
    result_OAA = OAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.01, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True, with_CAA_initialization = False)

    A_OAA = result_OAA.A
    Z_OAA = result_OAA.Z

    result_RBOAA = RBOAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.01, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True)
    A_RBOAA = result_RBOAA.A
    Z_RBOAA = result_RBOAA.Z
    
    # NMI
    NMI_OAA.append(NMI(A_true, A_OAA))
    NMI_RBOAA.append(NMI(A_true, A_RBOAA))
    
    # with Z true in the ordinal domain
    MCC_OAA.append(MCC(Z_true, Z_OAA))
    MCC_RBOAA.append(MCC(Z_true, Z_RBOAA))
    
    # with Z true in the alpha domain
    MCC_OAA_alpha.append(MCC(Z_true_alpha, Z_OAA))
    MCC_RBOAA_alpha.append(MCC(Z_true_alpha, Z_RBOAA))
    
    loss_f_OAA.append(result_OAA.loss[-1])
    loss_f_RBOAA.append(result_RBOAA.loss[-1])
    
    print("BACHELOR BØRGE MCC")
    MCC_OAA_BB.append( archetype_correlation(Z_true, Z_OAA)[2])
    MCC_RBOAA_BB.append( archetype_correlation(Z_true, Z_RBOAA)[2])
    
    





#%%
import matplotlib.pyplot as plt
def softplus(s):
    return np.log(1+np.exp(s))

sigmas_softplussed = [softplus(s) for s in sigmas]


print(MCC_OAA_BB)
print("MCC for RBOAA:")
print(MCC_RBOAA_BB)
#%%
plt.scatter(sigmas_softplussed, NMI_OAA, label = "OAA")
plt.plot(sigmas_softplussed, NMI_OAA, '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('NMI', fontsize=18)


plt.scatter(sigmas_softplussed, NMI_RBOAA, label = "RBOAA")
plt.plot(sigmas_softplussed, NMI_RBOAA, '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('NMI', fontsize=18)
plt.title('NMI plot for the two methods no RB', fontsize = 22)
plt.legend()
plt.show()



plt.scatter(sigmas_softplussed, MCC_OAA, label = "OAA")
plt.plot(sigmas_softplussed, MCC_OAA, '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)


plt.scatter(sigmas_softplussed, MCC_RBOAA, label = "RBOAA")
plt.plot(sigmas_softplussed, MCC_RBOAA, '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)
plt.title('MCC ordinal and alpha domain no RB', fontsize = 22)

plt.legend()
plt.show()



plt.scatter(sigmas_softplussed, MCC_OAA_alpha, label = "OAA")
plt.plot(sigmas_softplussed, MCC_OAA_alpha, '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)


plt.scatter(sigmas_softplussed, MCC_RBOAA_alpha, label = "RBOAA")
plt.plot(sigmas_softplussed, MCC_RBOAA_alpha, '--')
plt.xlabel('sigma', fontsize=18)
plt.ylabel('MCC', fontsize=18)
plt.title('MCC alpha domain no RB', fontsize = 22)

plt.legend()
plt.show()




plt.scatter(sigmas_softplussed, loss_f_OAA, label ="OAA")
plt.plot(sigmas_softplussed, loss_f_OAA)
plt.xlabel('sigma', fontsize=18)
plt.ylabel('loss', fontsize=18)
plt.scatter(sigmas_softplussed, loss_f_RBOAA, label = "RBOAA")
plt.plot(sigmas_softplussed, loss_f_RBOAA)
plt.xlabel('sigma', fontsize=18)
plt.ylabel('loss', fontsize=18)

plt.legend()
plt.title('Loss for increasing noise no RB', fontsize = 22)


#%%
# plots = _plots()
# # print(Z_true[0,:])
# plots._barplot_all(Z_true, columns = ["A"+str(i) for i in range(len(Z_true[0,:]))])
# plots._barplot_all(Z_OAA, columns = ["A"+str(i) for i in range(len(Z_OAA[0,:]))])
# plots._barplot_all(Z_RBOAA, columns = ["A"+str(i) for i in range(len(Z_RBOAA[0,:]))])



# print(Z_true, Z_OAA)
