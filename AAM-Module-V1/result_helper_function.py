
def result_helper_function(params):
    from AAM import AA
    
    N = 30
    M = 21
    K = 5
    p = 6
    rb = True
    n_iter = 8000
    reps = 10

    type = params[0]
    s = params[1]
    arch = params[2]
    a_param = params[3]
    b_param = params[4]
    
    AAM = AA()
    AAM.create_synthetic_data(N=N, M=M, K=K, p=p, sigma=s, rb=rb, a_param=a_param, b_param=b_param)
    for rep in range(reps):
        AAM.analyse(AA_type = type, with_synthetic_data = True, K=arch, n_iter = n_iter, mute=True)
        analysis_name = "sigma_" + str(s) + "_Arche_" + str(arch) + "_a_" + str(a_param) + "_b_" + str(b_param) + "_rep_" + str(rep)
        AAM.save_analysis(filename =  analysis_name, model_type = type, result_number = 0, with_synthetic_data=True) 
