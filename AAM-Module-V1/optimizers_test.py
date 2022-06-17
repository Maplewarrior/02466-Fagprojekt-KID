from AAM import AA
from synthetic_data_class import _synthetic_data
from plots_class import _plots
import numpy as np
import pandas as pd

from eval_measures import NMI
from eval_measures import MCC
from eval_measures import BDM

from OAA_class import _OAA
from RBOAA_class import _RBOAA
from TSAA_class import _TSAA

AAM = AA()

AAM.create_synthetic_data(N = 10000, M=21,K=5,sigma=-20,a_param=1,b_param=1000, mute=True)
reps = 10
AA_types = ["CAA","TSAA","RBOAA", "OAA"]
losses = {"CAA":[],"TSAA":[],"RBOAA":[],"OAA":[]}

for AA_type in AA_types:
    print(AA_type)
    for rep in range(reps):
        AAM.analyse(5,6,2000,True,AA_type,0.01,False,True,True)
        if AA_type in ["TSAA","CAA"]:
            losses[AA_type].append(AAM._synthetic_results[AA_type][0].RSS[0])
        else:
            losses[AA_type].append(AAM._synthetic_results[AA_type][0].loss[0])
        print(rep)

dataframe = pd.DataFrame.from_dict(losses)
csv_name = 'optimizers results/optimizers_test_SGD.csv'
dataframe.to_csv(csv_name, index=False) 