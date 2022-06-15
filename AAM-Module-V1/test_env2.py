from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np
from AAordinalSampler import OrdinalSampler

from eval_measures import NMI
from eval_measures import MCC
from eval_measures import BDM

from OAA_class import _OAA
from RBOAA_class import _RBOAA



AAM = AA()

AAM.create_synthetic_data(1000,21,5,6,-3,True,1,1000,True,0,False)
Xnew,Xhat,archetypes,S,beta = OrdinalSampler(1000,21,[1,2,3,4,5,6],0.04858735157,5,1,1000,True)

AAM.analyse(5,6,2000,True,"RBOAA",0.05,False,True,True,False)
restult1_RBOAA = AAM._synthetic_results["RBOAA"][0]
AAM.analyse(5,6,2000,True,"OAA",0.01,False,True,True,False)
restult1_OAA = AAM._synthetic_results["OAA"][0]

OAA = _OAA()
resultOAA = OAA._compute_archetypes(Xnew.T,5,6,2000,0.05,False,[],True,True,True,False,False)

RBOAA = _RBOAA()
resultRBOAA = RBOAA._compute_archetypes(Xnew.T,5,6,2000,0.01,False,[],True,True,True,False)

print("OURS")
print(NMI(AAM._synthetic_data.A,restult1_RBOAA.A))
print(NMI(AAM._synthetic_data.A,restult1_OAA.A))
print("BB")
print(NMI(S.T,resultRBOAA.A))
print(NMI(S.T,resultOAA.A))