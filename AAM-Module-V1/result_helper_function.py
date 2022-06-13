

from telnetlib import OLD_ENVIRON
from eval_measures import calcMI


def result_helper_function2(params):
    from AAM import AA
    import numpy as np
    import pandas as pd
    from eval_measures import NMI
    from eval_measures import MCC
    from eval_measures import BDM
    
    N = 5000
    M = 21
    K = 5
    p = 6
    rb = True
    n_iter = 2000

    #reps = 10
    reps = 2
    #analysis_archs = np.arange(3,11)
    analysis_archs = [5]
    AA_types = ["CAA","RBOAA", "OAA"]

    s = params[0]
    synthetic_arch = params[1]
    a_param = params[2]
    b_param = params[3]
    
    AA_types_list = []
    analysis_archs_list = []
    reps_list = []
    losses_list = []
    NMIs_list = []
    MCCs_list = []
    BDM_list = []

    AAM = AA()

    AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=rb, a_param=a_param, b_param=b_param,mute=True)
    syn_A = AAM._synthetic_data.A
    syn_Z = AAM._synthetic_data.Z
    syn_betas = AAM._synthetic_data.betas

    done = False

    for AA_type in AA_types:
        if AA_type == "CAA":
            lr = 0.01
        elif AA_type == "TSAA":
            lr = 0.01
        elif AA_type == "OAA":
            lr = 0.01
        elif AA_type == "RBOAA":
            lr = 0.01
        for analysis_arch in analysis_archs:
            for rep in range(reps):

                AA_types_list.append(AA_type)
                analysis_archs_list.append(analysis_arch)
                reps_list.append(rep)

                if AA_type == "OAA":
                    AAM.analyse(AA_type = AA_type, lr=lr, with_synthetic_data = True, K=analysis_arch, n_iter = n_iter, mute=False, early_stopping=True, with_CAA_initialization=True, p=p)
                elif AA_type == "RBOAA":
                    AAM.analyse(AA_type = AA_type, lr=lr, with_synthetic_data = True, K=analysis_arch, n_iter = n_iter, mute=False, early_stopping=True, with_CAA_initialization=True,p=p)
                else:
                    AAM.analyse(AA_type = AA_type, lr=lr, with_synthetic_data = True, K=analysis_arch, n_iter = n_iter, mute=False, early_stopping=True)

                analysis_A = AAM._synthetic_results[AA_type][0].A

                if AA_type == "RBOAA":
                    print("A ANAL")
                    print(analysis_A[0,0])
                    print("A TRUE")
                    print(syn_A[0,0])

                if AA_type in ["CAA","TSAA"]:
                    analysis_Z = AAM._synthetic_results[AA_type][0].Z
                    loss = AAM._synthetic_results[AA_type][0].RSS[-1]
                    BDM_list.append("NaN")
                else:
                    analysis_Z = AAM._synthetic_results[AA_type][0].Z
                    loss = AAM._synthetic_results[AA_type][0].loss[-1]
                    analysis_betas = AAM._synthetic_results[AA_type][0].b
                    BDM_list.append(BDM(syn_betas,analysis_betas,AA_type))
                
                losses_list.append(loss)
                NMIs_list.append(NMI(analysis_A,syn_A))
                MCCs_list.append(MCC(analysis_Z,syn_Z))


    dataframe = pd.DataFrame.from_dict({
        'sigma': s,
        'synthetic_k': synthetic_arch,
        'a_param': a_param,
        'b_param': a_param,
        'AA_type': AA_types_list, 
        'analysis_k': analysis_archs_list, 
        'rep': reps_list, 
        'loss': losses_list, 
        'NMI': NMIs_list, 
        'MCC': MCCs_list,
        'BDM': BDM_list})

    csv_name = 'result dataframes/' + str(s) + "_" + str(synthetic_arch) + "_" + str(a_param) + "_" + str(b_param) + "HEY" + ".csv"
    dataframe.to_csv(csv_name, index=False) 
