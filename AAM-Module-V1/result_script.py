
if __name__ ==  '__main__':
    from result_helper_function import result_helper_function
    from ast import arg
    import multiprocessing

    #### ONLY CHANGE THIS VARIABLE ####
    archetypes = [2,3,4]
    # archetypes = [5,6,7]
    # archetypes = [8,9,10]

    a_param = [2]
    sigma_vals = [-2.25]
    b_param = [10]
    sigma_stds = [0.5]
    varying_archetypes = True
    l = []

    for sigma in sigma_vals:
        for arch in archetypes:
            for a in a_param:
                for b in b_param:
                    for sigma_std in sigma_stds:
                        l.append([sigma,arch,a,b,varying_archetypes,sigma_std])

    
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(result_helper_function, l)
