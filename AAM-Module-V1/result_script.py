
if __name__ ==  '__main__':
    from result_helper_function import result_helper_function
    from ast import arg
    import multiprocessing

    
    AA_types = ["RBOAA", "CAA", "OAA",  "TSAA"]
    sigma_vals = [-20, -1.26, -0.43]
    archetypes = [3,5,7]
    a_param = [0.85, 1, 2]
    b_param = [1, 10, 1000]
    l = []
    for type in AA_types:
        for sigma in sigma_vals:
            for arch in archetypes:
                for a in a_param:
                    for b in b_param:
                        l.append([type,sigma,arch,a,b])

    
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(result_helper_function, l)

