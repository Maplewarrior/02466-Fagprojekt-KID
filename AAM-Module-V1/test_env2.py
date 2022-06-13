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

AAM = AA()
AAM.create_synthetic_data(5000,21,5,6,-3.25,True,10,1,True,sigma_std=0.5)

AAM.analyse(5,6,10000,True,"RBOAA",0.025,False,True,True)
print(BDM(AAM._synthetic_data.betas,AAM._synthetic_results["RBOAA"][0].b, "RBOAA"))

print(AAM._synthetic_data.betas)
print(AAM._synthetic_results["RBOAA"][0].b)

AAM.analyse(5,6,10000,True,"OAA",0.025,False,True,True)
print(BDM(AAM._synthetic_data.betas,AAM._synthetic_results["OAA"][0].b, "OAA"))