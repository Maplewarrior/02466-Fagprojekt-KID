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
from TwoStepOAA import _RBOAA as _twoStepRBOAA

NMI_TWO = []
NMI_C = []
NMI_OAA = []

MCC_TWO = []
MCC_TWO_tilde = []
MCC_C = []
MCC_OAA = []

B_TWO = []
B_C = []
B_OAA = []

SIGMA_C = []
SIGMA_TWO = []
SIGMA_OAA = []



for i in range(1):
    AAM = AA()
    AAM.create_synthetic_data(N=5000,M=21,K=5,p=6,sigma=-2.25,rb=True,b_param=1,a_param=1,mute = True,sigma_std=0)

    data = AAM._synthetic_data

    OAA = _OAA()
    result_OAA = OAA._compute_archetypes(data.X,data.K,data.p,10000,0.05,False,[],True,True,True)

    RBOAA = _RBOAA()
    result = RBOAA._compute_archetypes(data.X,data.K,data.p,10000,0.025,False,[],True,True,True)

    RBOAATWO = _twoStepRBOAA()
    result2 = RBOAATWO._compute_archetypes(data.X,data.K,data.p,10000,0.025,False,[],True,True,True)

    NMI_C.append(NMI(result.A,data.A))
    NMI_TWO.append(NMI(result2.A,data.A))
    NMI_OAA.append(NMI(result_OAA.A,data.A))

    MCC_C.append(MCC(result.Z,data.Z))
    MCC_TWO.append((MCC(result2.Z,data.Z)))
    MCC_TWO_tilde.append((MCC(result2.Z_tilde,data.Z)))
    MCC_OAA.append((MCC(result_OAA.Z,data.Z)))

    B_C.append(BDM(data.betas,result.b,"RBOAA"))
    B_TWO.append((BDM(data.betas,result2.b,"RBOAA")))
    B_OAA.append((BDM(data.betas,result_OAA.b,"OAA")))

    SIGMA_C.append(result.sigma)
    SIGMA_TWO.append(result2.sigma)
    SIGMA_OAA.append(result_OAA.sigma)

    print(i)

print("MCC MEAN")

print(np.mean(MCC_C))
print(np.mean(MCC_TWO))
print(np.mean(MCC_OAA))

print("MCC MAX")

print(np.max(MCC_C))
print(np.max(MCC_TWO))
print(np.max(MCC_OAA))

print("NMI MEAN")

print(np.mean(NMI_C))
print(np.mean(NMI_TWO))
print(np.mean(NMI_OAA))

print("NMI MAX")

print(np.max(NMI_C))
print(np.max(NMI_TWO))
print(np.max(NMI_OAA))

print("BOUNDARIES MEAN")

print(np.mean(B_C))
print(np.mean(B_TWO))
print(np.mean(B_OAA))

print("BOUNDARIES MIN")

print(np.min(B_C))
print(np.min(B_TWO))
print(np.min(B_OAA))

print("SIGMA MEAN")

print(np.mean(SIGMA_C))
print(np.mean(SIGMA_TWO))
print(np.mean(SIGMA_OAA))