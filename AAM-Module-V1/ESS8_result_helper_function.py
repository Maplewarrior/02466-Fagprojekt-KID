

def ESS8_result_helper_function(params):
    from AAM import AA
    from eval_measures import NMI
    from eval_measures import MCC
    from eval_measures import BDM
    import numpy as np
    import pandas as pd

    AA_type = params[0]
    k = params[1]
    reps = 2
    if AA_type == "CAA":
        lr = 0.1
    elif AA_type == "TSAA":
        lr = 0.01
    elif AA_type == "OAA":
        lr = 0.01
    elif AA_type == "RBOAA":
        lr = 0.01

    A_matricies = []
    Z_matricies = []
    boundaries = []
    losses = []

    AAM = AA()
    AAM.load_csv("ESS8_data.csv",np.arange(12,33),mute=True)

    for rep in range(reps):
        print("{0} and {1}".format(AA_type,rep))
        AAM.analyse(k,6,4000,True,AA_type,lr,True,False,True)
        result = AAM._results[AA_type][0]

        A_matricies.append(result.A)
        Z_matricies.append(result.Z)

        losses.append(result.loss[0])
        if AA_type in ["OAA","RBOAA"]:
            boundaries.append(result.b)
    
    NMI_list = []
    MCC_list = []
    BDM_list = []


    for i in range(reps):
        for j in range(reps):
            if not i == j:
                NMI_list.append(NMI(A_matricies[i],A_matricies[j]))
                MCC_list.append(MCC(Z_matricies[i],Z_matricies[j]))
                if AA_type in ["OAA","RBOAA"]:
                    BDM_list.append(BDM(boundaries[i],boundaries[j],AA_type))


    mean_NMI = [np.mean(NMI_list)]
    mean_MCC = [np.mean(MCC_list)]

    max_NMI = [np.max(NMI_list)]
    max_MCC = [np.max(MCC_list)]

    min_loss = [np.min(losses)]
    mean_loss = [np.mean(losses)]

    if AA_type in ["OAA","RBOAA"]:
        mean_BDM = [np.mean(BDM_list)]
        max_BDM = [np.max(BDM_list)]

    if AA_type in ["OAA","RBOAA"]:
        dataframe = pd.DataFrame.from_dict({
            'mean NMI': mean_NMI,
            'mean MCC': mean_MCC,
            'mean BDM': mean_BDM,
            'mean loss': mean_loss,
            'max NMI': max_NMI,
            'max MCC': max_MCC,
            'max BDM': max_BDM,
            'min loss': min_loss})
    else:
        dataframe = pd.DataFrame.from_dict({
            'mean NMI': mean_NMI,
            'mean MCC': mean_MCC,
            'mean loss': mean_loss,
            'max NMI': max_NMI,
            'max MCC': max_MCC,
            'min loss': min_loss})
    
    csv_name = "ESS8 results/AA_ESS8_{0}_K={1}".format(AA_type,k)
    dataframe.to_csv(csv_name, index=False) 
    
