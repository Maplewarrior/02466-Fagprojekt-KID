from telnetlib import OLD_ENVIRON
from eval_measures import calcMI


def result_helper_function_model_stability(params):
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
    reps = 30
    AA_types = ["CAA", "TSAA", "RBOAA", "OAA"]

    s = params[0]
    a_param = params[2]
    b_param = params[3]
    sigma_std = params[5]
    synthetic_arch = params[1]
    
    if params[4]:
        analysis_archs = np.arange(2,11)
    else:
        analysis_archs = [5]

    AA_types_list = []
    analysis_archs_list = []
    reps_list = []
    losses_list = []
    sigma_est_list = []
    A_MSMD = []
    Z_MSMD = []
    beta_MSMD = []

    AAM = AA()
    if b_param == "RB_false":
        AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=False, a_param=a_param, b_param=0,mute=True, sigma_std=sigma_std)
    else:
        AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=True, a_param=a_param, b_param=b_param,mute=True, sigma_std=sigma_std)

    for AA_type in AA_types:
        if AA_type == "CAA":
            lr = 0.05
        elif AA_type == "TSAA":
            lr = 0.01
        elif AA_type == "OAA":
            lr = 0.05
        elif AA_type == "RBOAA":
            lr = 0.025

        A_list = []
        Z_list = []
        beta_list = []
        
        for analysis_arch in analysis_archs:
            for rep in range(reps):

                AAM.analyse(AA_type = AA_type, lr=lr, with_synthetic_data = True, mute=True, K=analysis_arch, n_iter = n_iter, early_stopping=True, with_hot_start=True, p=p)
                A_list.append(AAM._synthetic_results[AA_type][0].A)
                Z_list.append(AAM._synthetic_results[AA_type][0].Z)
                if AA_type in ["OAA","RBOAA"]:
                    beta_list.append(AAM._synthetic_results[AA_type][0].b)
                else:
                    beta_list.append("NaN")

        AA_types_list.append(AA_type)
        if AA_type in ["OAA"]:
            loss = AAM._synthetic_results[AA_type][0].loss[-1]
            beta_MSMD.append(MSMD(beta_list)[0])
        elif AA_type in ["RBOAA"]:
            loss = AAM._synthetic_results[AA_type][0].loss[-1]
            beta_MSMD.append(MSMD(beta_list))
        else:
            beta_MSMD.append("NaN")
            loss = AAM._synthetic_results[AA_type][0].RSS[-1]
        losses_list.append(loss)
        Z_MSMD.append(MSMD(Z_list))
        A_MSMD.append(MSMD(A_list))
        analysis_archs_list.append(analysis_arch)
        reps_list.append("NaN")
        sigma_est_list.append("NaN")

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
        'Est. sigma': sigma_est_list,
        'AA_type': AA_types_list, 
        'A_SDM': A_MSMD,
        'Z_SDM': Z_MSMD,
        'beta_SDM': beta_MSMD})

    csv_name = 'result stability/' + str(s) + "_" + str(sigma_std) + "_" + str(synthetic_arch) + "_" + str(a_param) + "_" + str(b_param) + "_" + str(synthetic_arch) + ".csv"
    dataframe.to_csv(csv_name, index=False) 
