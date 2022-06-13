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
AAM.create_synthetic_data(1000,21,5,6,-2.25,False,10,1,True,sigma_std=0.5)

# plotter = _plots()
# plotter._barplot_all(AAM._synthetic_data.Z,AAM._synthetic_data.columns)

AAM.analyse(5,6,10000,True,"OAA",0.025,False,True,True)

# AAM.plot("RBOAA","loss_plot",with_synthetic_data=True)
# AAM.plot("RBOAA","barplot_all",with_synthetic_data=True)


# print(AAM._synthetic_data.betas)
# print(AAM._synthetic_results["RBOAA"][0].b)

# print(MCC(AAM._synthetic_results["RBOAA"][0].Z,AAM._synthetic_data.Z))
# print(NMI(AAM._synthetic_results["RBOAA"][0].A,AAM._synthetic_data.A))
print(BDM(AAM._synthetic_data.betas,AAM._synthetic_results["OAA"][0].b, "OAA"))