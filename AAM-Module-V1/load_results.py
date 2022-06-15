import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 

### sigma,sigma_std,synthetic_k,a_param,b_param,AA_type,analysis_k,rep,loss,NMI,MCC,BDM,Est. sigma

results = {
    'sigma': np.array([]),
    'sigma_std': np.array([]),
    'synthetic_k': np.array([]),
    'a_param': np.array([]),
    'b_param': np.array([]),
    'AA_type': np.array([]),
    'analysis_k': np.array([]),
    'rep': np.array([]),
    'loss': np.array([]),
    'NMI': np.array([]),
    'MCC': np.array([]),
    'BDM': np.array([]),
    'Est. sigma': np.array([]),
}

# sigma,sigma_std,synthetic_k,a_param,b_param,AA_type,analysis_k,rep,loss,Est. sigma,A_SDM,Z_SDM,beta_SDM

results2 = {
    'sigma': np.array([]),
    'sigma_std': np.array([]),
    'synthetic_k': np.array([]),
    'a_param': np.array([]),
    'b_param': np.array([]),
    'AA_type': np.array([]),
    'analysis_k': np.array([]),
    'rep': np.array([]),
    'loss': np.array([]),
    'Est. sigma': np.array([]),
    'A_SDM': np.array([]),
    'Z_SDM': np.array([]),
    'beta_SDM': np.array([])
}

# sigma,sigma_std,synthetic_k,a_param,b_param,AA_type,analysis_k,rep,loss,NMI,MCC,BDM,Est. sigma,centralization

results3 = {
    'sigma': np.array([]),
    'sigma_std': np.array([]),
    'synthetic_k': np.array([]),
    'a_param': np.array([]),
    'b_param': np.array([]),
    'AA_type': np.array([]),
    'analysis_k': np.array([]),
    'rep': np.array([]),
    'loss': np.array([]),
    'NMI': np.array([]),
    'MCC': np.array([]),
    'BDM': np.array([]),
    'Est. sigma': np.array([]),
    'centralization': np.array([])
}


directory = 'centralization results'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath) and not filename == ".DS_Store":
        file = open(filepath,'rb')
        data = pd.read_csv(filepath,encoding = "ISO-8859-1")


        for key in list(results2.keys()):
            for i in range(len(data[key])):
                results2[key] = np.append(results2[key], data[key][i])

dataframe = pd.DataFrame.from_dict(results2)
csv_name = "full_centralization_dataset.csv"
dataframe.to_csv(csv_name, index=False) 

