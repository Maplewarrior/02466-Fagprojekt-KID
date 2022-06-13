from tkinter import W
from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np

from eval_measures import NMI
from eval_measures import MCC

from OAA_class import _OAA
from RBOAA_class import _RBOAA


AAM = AA()
AAM.create_synthetic_data(40000,5,3,6,-4.6,True,10,1,False)

plotter = _plots()
plotter._barplot_all(AAM._synthetic_data.Z,AAM._synthetic_data.columns)

AAM.analyse(3,6,10000,True,"OAA",0.001,False,True,True)

AAM.plot("OAA","loss_plot",with_synthetic_data=True)
AAM.plot("OAA","barplot_all",with_synthetic_data=True)

print(MCC(AAM._synthetic_results["OAA"][0].Z,AAM._synthetic_data.Z))
print(NMI(AAM._synthetic_results["OAA"][0].A,AAM._synthetic_data.A))