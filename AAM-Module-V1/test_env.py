from AAM import AA
from synthetic_data_class import _synthetic_data
import numpy as np
from eval_measures import NMI
from eval_measures import MCC
from RBOAA_classOLDVERSION import _RBOAA


AAM = AA()

#AAM.load_csv("ESS8_data.csv",range(12,33), rows = 1000)

AAM.create_synthetic_data(N = 2000, M=21,K=5,sigma=-4,a_param=1,b_param=1000, rb=False,p=6)
synthetic_data = AAM._synthetic_data
synthetic_dataX = synthetic_data.X
synthetic_dataA = synthetic_data.A
synthetic_dataZ = synthetic_data.Z

AAM.analyse(AA_type = "RBOAA", with_synthetic_data=True, K=5,lr=0.001, n_iter=1000, early_stopping=False,with_CAA_initialization=False, p=6)
resultA = AAM._synthetic_results["RBOAA"][0].A
resultZ = AAM._synthetic_results["RBOAA"][0].Z

print(NMI(resultA,synthetic_dataA))
print(MCC(resultZ,synthetic_dataZ))

analysis_A = AAM._synthetic_results["RBOAA"][0].A
syn_A = AAM._synthetic_data.A

RBOAA = _RBOAA()
result = RBOAA._compute_archetypes(synthetic_dataX, 5, 1000,0.001,False,[],True)
resultA = result.A
resultZ = result.Z

print(NMI(resultA,synthetic_dataA))
print(MCC(resultZ,synthetic_dataZ))
