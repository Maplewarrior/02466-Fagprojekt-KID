from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np

from eval_measures import NMI
from eval_measures import MCC
from eval_measures import BDM

from OAA_class import _OAA
from RBOAA_class import _RBOAA
from TSAA_class import _TSAA

AAM = AA()

AAM.create_synthetic_data(1000,21,5,6,-20,False,1,1,False,0)

AAM.analyse(10,6,2000,False,"RBOAA",0.01,False,True,True)
print(AAM._synthetic_results["RBOAA"][0].loss)
AAM.plot("RBOAA","loss_plot",with_synthetic_data=True)