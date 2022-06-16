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

AAM.create_synthetic_data(N = 10000, M=21,K=5,sigma=-20,a_param=1,b_param=1000, mute=True)
reps = 10

losses = []

for rep in range(reps):
    AAM.analyse(5,6,5000,True,"RBOAA",0.01,False,False,True)
    AAM.plot("RBOAA","attribute_scatter_plot",with_synthetic_data=False)    
