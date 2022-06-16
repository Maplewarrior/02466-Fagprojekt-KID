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
# AAM.create_synthetic_data(40000,10,5,6,-100,False,5,1,False,0)
AAM.load_csv("ESS8_data.csv",np.arange(13,25),100)
AAM.analyse(5,6,5000,True,"RBOAA",0.01,False,False,True)
AAM.plot("RBOAA","attribute_scatter_plot",with_synthetic_data=False)