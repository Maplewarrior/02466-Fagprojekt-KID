
def result_helper_function(params):
    from AAM import AA
    import numpy as np
    
    N = 5000
    M = 21
    K = 5
    p = 6
    rb = True
    n_iter = 10000
    n_iter = 100
    reps = 5
    analysis_archs = np.arange(2,11,2)

    AA_types = ["RBOAA", "CAA", "OAA",  "TSAA"]
    s = params[0]
    synthetic_arch = params[1]
    a_param = params[2]
    b_param = params[3]
    

    AAM = AA()
    AAM.create_synthetic_data(N=N, M=M, K=synthetic_arch, p=p, sigma=s, rb=rb, a_param=a_param, b_param=b_param)
    for AA_type in AA_types:
        for analysis_arch in analysis_archs:
            for rep in range(reps):
                AAM.analyse(AA_type = AA_type, with_synthetic_data = True, K=analysis_arch, n_iter = n_iter, mute=True, early_stopping=True)
                analysis_name = "sigma_" + str(s) + "_Arche_" + str(analysis_arch) + "_a_" + str(a_param) + "_b_" + str(b_param) + "_rep_" + str(rep) + "_SYNARCH_" + str(synthetic_arch)
                if AA_type == AA_types[0] and analysis_arch == analysis_archs[0] and rep == 0:
                    AAM.save_analysis(filename =  analysis_name, model_type = AA_type, result_number = 0, with_synthetic_data=True, save_synthetic_data=True)
                else:
                    AAM.save_analysis(filename =  analysis_name, model_type = AA_type, result_number = 0, with_synthetic_data=True, save_synthetic_data=False)

