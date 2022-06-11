from pickle import FALSE
from statistics import mode
from webbrowser import Galeon
from AAM import AA
from synthetic_data_class import _synthetic_data
import numpy as np
from eval_measures import NMI
from eval_measures import MCC
from eval_measures import BDM
from RBOAA_class import _RBOAA
from plots_class import _plots
from OAA_class import _OAA
from AAordinalSampler import OrdinalSampler
import torch



rb = False
betaParm = 1000
K = 4
N = 10000
M = 21
epokes = 1000
sigma = -1000
p = 5
a_param = 1


data = _synthetic_data(N=N, M=M ,K=K, p=p, sigma=sigma, rb=rb, a_param = a_param, b_param = betaParm)
X = data.X
A_true = data.A
# OAA = _OAA()

# result = OAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.005, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True, with_CAA_initialization = False)
# result._plot("loss_plot",["1","2","3","4","5"],1,1,1)


RBOAA = _RBOAA()
result = RBOAA._compute_archetypes(X=X, K=K, p=p, n_iter=10000, lr=0.01, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True)
#%%
b = torch.rand((N, p-1))
print(b.shape)
pad = torch.nn.ConstantPad1d(1, 0)
bn = pad(b)
bn[:,-1]=1.0
print(bn.shape)
print(bn[:10,0],  bn[:10,-1])
#%%

# Xnew,Xhat,archetypes,A_true,beta = OrdinalSampler(N=5000,M=5,alphaFalse=[1,2,3,4,5],sigma=0.01,K=5,achetypeParm=1,betaParm=1,bias=False)
# print(beta)
# OAA = _OAA()
# result = OAA._compute_archetypes(Xnew.T,5,5,100,0.01,False,["1","2","3","4","5"],True,True,False)


# result._plot("barplot_all",["1","2","3","4","5"],1,1,1)

# print(BDM(beta[:,1:-1],result.b, "OAA"))
# print(result.Z)
# print(archetypes.T)

# # print(NMI(result.A.T,A_true))
# print(NMI(result.Z,archetypes.T))

# print(beta)


# b = torch.tensor([1,2,3])
# print(b)
# print(torch.cat((torch.tensor([0.0]),b,torch.tensor([1.0]))))

# AAM = AA()
# AAM.create_synthetic_data(N=5000, M=5, K=3, p=6, sigma=-20, rb=False, a_param=-5, b_param=1,mute=True)
# AAM.analyse(AA_type = "OAA", lr=0.01, with_synthetic_data = True, K=3, n_iter = 2000, mute=False, early_stopping=False, with_CAA_initialization=False, p=6)

# AAM.plot(model_type="OAA",plot_type="loss_plot", with_synthetic_data=True)
# AAM.plot(model_type="OAA",plot_type="barplot_all", with_synthetic_data=True)

# data = AAM._synthetic_data

# print(archetypes)
# result._plot("barplot_all",1,1,1,1)
# # result._plot("loss_plot",1,1,1,1)
# plotter = _plots()
# plotter._barplot_all(data.Z, ["1","2","3","4","5"])


# print(AAM._synthetic_results["OAA"][0].Z)
# print(data.Z)

# print(NMI(data.Z,AAM._synthetic_results["OAA"][0].Z))
# print(MCC(data.Z,AAM._synthetic_results["OAA"][0].Z))
