from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np

from eval_measures import NMI
from eval_measures import MCC

from OAA_class import _OAA
from RBOAA_class import _RBOAA
from TSAA_class import _TSAA

AAM = AA()

AAM.create_synthetic_data(1000,10,5,6,-20,True,2,1,False,0)

AAM.analyse(5,6,10000,True,"OAA",0.01,False,True,True)

AAM.analyse(5,6,10000,True,"RBOAA",0.01,False,True,True)

print(AAM._synthetic_data.A.shape)
print(AAM._synthetic_results["OAA"][0].A.shape)

print(NMI(AAM._synthetic_results["OAA"][0].A.T,AAM._synthetic_data.A))
print(NMI(AAM._synthetic_data.A,AAM._synthetic_results["RBOAA"][0].A.T))