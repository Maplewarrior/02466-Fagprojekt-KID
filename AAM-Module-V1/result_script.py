
if __name__ ==  '__main__':
    from result_helper_function import result_helper_function
    from result_helper_function_model_stability import result_helper_function_model_stability
    from result_helper_function_centralization import result_helper_function_centralization
    from ast import arg
    import multiprocessing

    #### ONLY CHANGE THIS VARIABLE ####
    #a_param = [0.85,1,2]
    a_param = [1]

    ## VARYING ARCHETYPES PAR ##
    # archetypes = [3,5,7]
    # sigma_vals = [-2.25]
    # b_param = [5]
    # sigma_stds = [0]
    # varying_archetypes = True

    ## MODEL STABILITY ##
    archetypes = [5]
    sigma_vals = [-2.25]
    b_param = [5]
    sigma_stds = [0]
    varying_archetypes = False

    ## REGULAR PARAMETRES ##
    # archetypes = [5]
    # sigma_vals = [-100,-2.97,-2.25,-1.82,-1.5,-1.259,-1.05]
    # b_param = [1,5,10,"RB_false"]
    # sigma_stds = [0,1]
    # varying_archetypes = False
    l = []

    for sigma in sigma_vals:
        for arch in archetypes:
            for a in a_param:
                for b in b_param:
                    for sigma_std in sigma_stds:
                        l.append([sigma,arch,a,b,varying_archetypes,sigma_std])

    
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        #p.map(result_helper_function_centralization, l)
        p.map(result_helper_function_model_stability, l)
        #p.map(result_helper_function, l)
