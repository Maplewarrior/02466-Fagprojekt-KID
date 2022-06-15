

if __name__ ==  '__main__':
    from AAM import AA
    from learning_rate_test_helper import learning_rate_test_helper_function
    import multiprocessing

    lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
    AA_types = ["CAA","TSAA","RBOAA","OAA"]
    dataframe_sizes = [1000,5000,10000]

    l = []
    for lr in lrs:
        for AA_type in AA_types:
            for dataframe_size in dataframe_sizes:
                l.append([lr,AA_type,dataframe_size])

    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(learning_rate_test_helper_function, l)
