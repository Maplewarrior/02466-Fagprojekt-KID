
# def result_helper_function(params):
#     from AAM import AA
#     import numpy as np
    
#     N = 10000
#     M = 21
#     K = 5
#     p = 6
#     rb = True
#     n_iter = 25000
#     reps = 10
#     analysis_archs = np.arange(2,11)

#     AA_types = ["RBOAA", "CAA", "OAA",  "TSAA"]
#     s = params[0]
#     synthetic_arch = params[1]
#     a_param = params[2]
#     b_param = params[3]
    

#     AAM = AA()
#     AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=rb, a_param=a_param, b_param=b_param)
#     for AA_type in AA_types:
#         for analysis_arch in analysis_archs:
#             for rep in range(reps):
#                 AAM.analyse(AA_type = AA_type, lr=0.05, with_synthetic_data = True, K=analysis_arch, n_iter = n_iter, mute=True, early_stopping=True)
#                 analysis_name = "sigma_" + str(s) + "_Arche_" + str(analysis_arch) + "_a_" + str(a_param) + "_b_" + str(b_param) + "_rep_" + str(rep) + "_SYNARCH_" + str(synthetic_arch)
#                 if AA_type == AA_types[0] and analysis_arch == analysis_archs[0] and rep == 0:
#                     AAM.save_analysis(filename =  analysis_name, model_type = AA_type, result_number = 0, with_synthetic_data=True, save_synthetic_data=True)
#                 else:
#                     AAM.save_analysis(filename =  analysis_name, model_type = AA_type, result_number = 0, with_synthetic_data=True, save_synthetic_data=False)




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
    reps = 1
    #analysis_archs = np.arange(3,11)
    analysis_archs = [5]
    AA_types = ["RBOAA", "CAA", "OAA",  "TSAA"]

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
            lr = 0.05
        elif AA_type == "OAA":
            lr = 0.05
        else:
            lr = 0.1
        for analysis_arch in analysis_archs:
            for rep in range(reps):

                AA_types_list.append(AA_type)
                analysis_archs_list.append(analysis_arch)
                reps_list.append(rep)

                AAM.analyse(AA_type = AA_type, lr=lr, with_synthetic_data = True, K=analysis_arch, n_iter = n_iter, mute=False, early_stopping=True)

                analysis_A = AAM._synthetic_results[AA_type][0].A

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
                NMIs_list.append(NMI(syn_A,analysis_A))
                MCCs_list.append(MCC(syn_Z,analysis_Z))

                print(MCC(syn_Z,analysis_Z))
                print(NMI(syn_A,analysis_A))


    dataframe = pd.DataFrame.from_dict({'sigma': s,
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

    print(NMIs_list)

    csv_name = 'result dataframes/' + str(s) + "_" + str(synthetic_arch) + "_" + str(a_param) + "_" + str(b_param) + "HEY" + ".csv"
    dataframe.to_csv(csv_name, index=False) 
