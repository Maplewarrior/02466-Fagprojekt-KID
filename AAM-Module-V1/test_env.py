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

AAM.load_csv("ESS8_data.csv",np.arange(12,33),1000)
AAM.analyse(10,6,10000,True,"RBOAA",0.01,False,False,True)
AAM.save_analysis("10 RBOAA",model_type="RBOAA")

# AAM.plot("RBOAA","loss_plot")
# AAM.plot("RBOAA","barplot",archetype_number=1)
# AAM.plot("RBOAA","barplot",archetype_number=2)
# AAM.plot("RBOAA","barplot",archetype_number=3)
# AAM.plot("RBOAA","barplot",archetype_number=4)
# AAM.plot("RBOAA","barplot",archetype_number=5)
# AAM.plot("RBOAA","barplot",archetype_number=6)
# AAM.plot("RBOAA","barplot",archetype_number=7)
# AAM.plot("RBOAA","barplot",archetype_number=8)
# AAM.plot("RBOAA","barplot",archetype_number=9)
# AAM.plot("RBOAA","barplot",archetype_number=10)
