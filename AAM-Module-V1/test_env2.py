from tkinter import W
from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np

from eval_measures import NMI
from eval_measures import MCC
from eval_measures import BDM

from OAA_class import _OAA
from RBOAA_class import _RBOAA
from AAordinalSampler import OrdinalSampler


b_params = [1, 10]
# sigmas = [-100, -4.6, -2.97, -2.25, -1.5 -1.259, -1.05]
            # 0,  0.01   0.05,  0.1,   0.2  0.25    0.3
sigmas = [-100,-4.6]
          

NMI_OAA_our = []
MCC_OAA_our = []
NMI_RBOAA_our = []
MCC_RBOAA_our = []

NMI_OAA_BB = []
MCC_OAA_BB = []
NMI_RBOAA_BB = []
MCC_RBOAA_BB = []

for k, sigma in enumerate(sigmas):
    print("run nr:", k)
    N = 10000
    M = 21
    K = 4
    p = 5
    sigma = sigma
    rb=True
    a_param = 1
    b_param = 1
    sigma_std = 0.0
    
    syn = _synthetic_data(N, M, K, p, sigma, rb, a_param, b_param, sigma_std=sigma_std)
    X = syn.X
    A_true = syn.A
    Z_true = syn.Z
    
    RBOAA = _RBOAA()
    OAA = _OAA()
    result_RBOAA = RBOAA._compute_archetypes(X, K, p, n_iter=10000, lr=0.01, mute = False, columns=syn.columns, early_stopping=True, with_OAA_initialization=True)
    result_OAA = OAA._compute_archetypes(X, K, p, n_iter=10000, lr=0.01, mute=False, columns=syn.columns, early_stopping=True)
    
    A_RBOAA = result_RBOAA.A
    Z_RBOAA = result_RBOAA.Z
    A_OAA = result_OAA.A
    Z_OAA = result_OAA.Z
    
    NMI_RBOAA_our.append(NMI(A_true, A_RBOAA))
    MCC_RBOAA_our.append(MCC(Z_true, Z_RBOAA))
    
    NMI_OAA_our.append(NMI(A_true, A_OAA))
    MCC_OAA_our.append(MCC(Z_true, Z_OAA))
    
    
    
    
    # alphaFalse = [1,2,3,4,5]
    # X_new, _, Z_BB, A_BB, _ = OrdinalSampler(N=N, M=M, alphaFalse=alphaFalse, sigma=np.log(1+np.exp(sigma)),K=K,achetypeParm=a_param,betaParm=b_param,bias=True)
    # X_new, Z_BB, A_BB = X_new.T, Z_BB.T, A_BB.T
    
    # print("Now for BB synthetic data")
    # RBOAA = _RBOAA()
    # OAA = _OAA()
    # result_RBOAA = RBOAA._compute_archetypes(X_new, K, p, n_iter=10000, lr=0.01, mute = True, columns=syn.columns, early_stopping=True, with_OAA_initialization=True)
    # result_OAA = OAA._compute_archetypes(X_new, K, p, n_iter=10000, lr=0.01, mute=True, columns=syn.columns, early_stopping=True)
    
    # A_RBOAA_B = result_RBOAA.A
    # Z_RBOAA_B = result_RBOAA.Z
    
    # A_OAA_B = result_OAA.A
    # Z_OAA_B = result_OAA.Z
    
    # NMI_RBOAA_BB.append(round(NMI(A_true, A_RBOAA_B),2))
    # MCC_RBOAA_BB.append(round(MCC(Z_true, Z_RBOAA_B),2))
    
    # NMI_OAA_BB.append(round(NMI(A_true, A_OAA_B),2))
    # MCC_OAA_BB.append(round(MCC(Z_true, Z_OAA_B),2))


#%%
print("NMI")
print("our OAA:", NMI_OAA_our)
print(print("our RBOAA: ", NMI_RBOAA_our))
# print("our OAA on BB:", NMI_OAA_BB)
# print(print("our RBOAA on BB: ", NMI_RBOAA_BB))

print("MCC")
print("our OAA:", MCC_OAA_our)
print(print("our RBOAA: ", MCC_RBOAA_our))
# print("our OAA on BB:", MCC_OAA_BB)
# print(print("our RBOAA on BB: ", MCC_RBOAA_BB))

print("Sigma vals", [np.log(1+np.exp(s)) for s in sigmas])


# AAM = AA()
# AAM.create_synthetic_data(N=10000,M=21,K=5,p=6,sigma=-3.25,rb=True,b_param=1000,a_param=0.95,mute = True,sigma_std=0.0)



# AAM.analyse(5,6,10000,True,"RBOAA",0.025,False,True,True)

#%%

# AAM.plot(model_type = "RBOAA", plot_type="barplot_all", with_synthetic_data=True)
# AAM.plot(model_type = "RBOAA", plot_type="PCA_scatter_plot", with_synthetic_data=True)
# AAM.plot(model_type = "RBOAA", plot_type = "mixture_plot", with_synthetic_data=True)
# print(BDM(AAM._synthetic_data.betas,AAM._synthetic_results["RBOAA"][0].b, "RBOAA"))

# print(AAM._synthetic_data.betas)
# print(AAM._synthetic_results["RBOAA"][0].b)

# AAM.analyse(5,6,10000,True,"OAA",0.025,False,True,True)
# print(BDM(AAM._synthetic_data.betas,AAM._synthetic_results["OAA"][0].b, "OAA"))