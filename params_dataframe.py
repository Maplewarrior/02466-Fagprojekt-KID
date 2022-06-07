import pandas as pd

AA_types = ["CAA","TSAA","OAA","RBOAA"]
k = [3,5,7]
sigma = [-20, -1.26, -0.43]
a_params = [0.85, 1, 2]
b_params = [1,10,1000]


Dataframe = []

for AA_type in AA_types:
    for k_param in k:
        for a_param in a_params:
            for b_param in b_params:
                Dataframe.append([AA_type,k_param,a_param,b_param])

pd_dataframe = pd.DataFrame(Dataframe)
pd_dataframe.to_csv("params.csv")