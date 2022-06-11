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

# data = _synthetic_data(1000,21,5,6,-4.6,False,1,1)
# dataX = data.X
# A_true = data.A
# OAA = _OAA()
# print(np.shape(dataX))
# result = OAA._compute_archetypes(dataX,5,6,1000,0.01,False,[],True,True,False)



Xnew,Xhat,archetypes,A_true,beta = OrdinalSampler(N=5000,M=5,alphaFalse=[1,2,3,4,5],sigma=0.01,K=5,achetypeParm=1,betaParm=1,bias=False)
print(beta)
OAA = _OAA()
result = OAA._compute_archetypes(Xnew.T,5,5,100,0.01,False,["1","2","3","4","5"],True,True,False)

result._plot("loss_plot",["1","2","3","4","5"],1,1,1)
result._plot("barplot_all",["1","2","3","4","5"],1,1,1)

print(BDM(beta[:,1:-1],result.b, "OAA"))
print(result.Z)
print(archetypes.T)

# print(NMI(result.A.T,A_true))
print(NMI(result.Z,archetypes.T))

print(beta)


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
