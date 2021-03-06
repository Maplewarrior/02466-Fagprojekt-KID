from telnetlib import OLD_ENVIRON
from eval_measures import calcMI


def result_helper_function_centralization(params):
    from AAM import AA
    import numpy as np
    import pandas as pd
    from eval_measures import NMI
    from eval_measures import MCC
    from eval_measures import BDM
    from eval_measures import MSMD
    
    N = 10000
    M = 21
    p = 6
    n_iter = 2000
    reps = 10
    AA_types = ["CAA", "TSAA", "RBOAA", "OAA"]

    s = params[0]
    a_param = params[2]
    b_param = params[3]
    sigma_std = params[5]
    synthetic_arch = params[1]
    analysis_arch = 5

    AA_types_list = []
    centralization_list = []
    analysis_archs_list = []
    reps_list = []
    losses_list = []
    sigma_est_list = []
    NMIs_list = []
    MCCs_list = []
    BDM_list = []

    AAM = AA()

    for AA_type in AA_types:
        if AA_type == "CAA":
            lr = 0.05
        elif AA_type == "TSAA":
            lr = 0.01
        elif AA_type == "OAA":
            lr = 0.05
        elif AA_type == "RBOAA":
            lr = 0.025
        
        for centralization in [True,False]:

            AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=True, a_param=a_param, b_param=b_param,mute=True, sigma_std=sigma_std, centralization=centralization)
            syn_A = AAM._synthetic_data.A
            syn_Z = AAM._synthetic_data.Z
            syn_betas = AAM._synthetic_data.betas

            for rep in range(reps):

                AA_types_list.append(AA_type)
                analysis_archs_list.append(analysis_arch)
                reps_list.append(rep)

                AAM.analyse(AA_type = AA_type, lr=lr, with_synthetic_data = True, mute=True, K=analysis_arch, n_iter = n_iter, early_stopping=True, with_hot_start=True, p=p, centralization=centralization)
                analysis_A = AAM._synthetic_results[AA_type][0].A
                analysis_Z = AAM._synthetic_results[AA_type][0].Z

                if AA_type in ["CAA","TSAA"]:
                    loss = AAM._synthetic_results[AA_type][0].RSS[-1]
                    BDM_list.append("NaN")
                    sigma_est_list.append("NaN")
                else:
                    loss = AAM._synthetic_results[AA_type][0].loss[-1]
                    analysis_betas = AAM._synthetic_results[AA_type][0].b
                    BDM_list.append(BDM(syn_betas,analysis_betas,AA_type))
                    sigma_est_list.append(np.mean(AAM._synthetic_results[AA_type][0].sigma))
                
                losses_list.append(loss)
                NMIs_list.append(NMI(analysis_A,syn_A))
                MCCs_list.append(MCC(analysis_Z,syn_Z))

                centralization_list.append(centralization)
                

    dataframe = pd.DataFrame.from_dict({
        'sigma': s,
        'sigma_std': sigma_std,
        'synthetic_k': synthetic_arch,
        'a_param': a_param,
        'b_param': b_param,
        'AA_type': AA_types_list, 
        'analysis_k': analysis_archs_list, 
        'rep': reps_list, 
        'loss': losses_list, 
        'NMI': NMIs_list, 
        'MCC': MCCs_list,
        'BDM': BDM_list,
        'Est. sigma': sigma_est_list,
        'AA_type': AA_types_list,
        'centralization': centralization_list})

    csv_name = 'centralization results/' + 'CENTRALIZATION_' + str(s) + "_" + str(sigma_std) + "_" + str(synthetic_arch) + "_" + str(a_param) + "_" + str(b_param) + "_" + str(synthetic_arch) + ".csv"
    dataframe.to_csv(csv_name, index=False) 
