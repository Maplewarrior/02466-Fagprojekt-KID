
if __name__ ==  '__main__':
    from result_helper_function import result_helper_function2
    from ast import arg
    import multiprocessing

    # ALL SIGMA:
    sigma_vals = [-20, -3, -2.25,-1.5,-1]
    #sigma_vals = [-4]
    #archetypes = [3,5,7]
    archetypes = [5]
    a_param = [0.85, 1, 2]
    # a_param = [1]
    b_param = [1, 10, 1000]
    #b_param = [1000]
    l = []

    for sigma in sigma_vals:
        for arch in archetypes:
            for a in a_param:
                for b in b_param:
                    l.append([sigma,arch,a,b])

    
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(result_helper_function2, l)

