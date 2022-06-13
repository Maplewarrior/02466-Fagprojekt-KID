

if __name__ ==  '__main__':
    from AAM import AA
    from learning_rate_test_helper import learning_rate_test_helper_function
    import multiprocessing

    #lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
    lrs = [0.05, 0.01, 0.005]
    #AA_types = ["CAA","TSAA","RBOAA","OAA"]
    AA_types = ["CAA","RBOAA","OAA"]

    l = []
    for lr in lrs:
        for AA_type in AA_types:
            l.append([lr,AA_type])

    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(learning_rate_test_helper_function, l)
