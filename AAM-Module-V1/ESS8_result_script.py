if __name__ ==  '__main__':
    import numpy as np
    from ESS8_result_helper_function import ESS8_result_helper_function
    import multiprocessing

    AA_types = ["CAA","RBOAA","OAA"]
    ks = np.arange(2,11)

    params = []

    for AA_type in AA_types:
        for k in ks:
            params.append([AA_type,k])

    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(ESS8_result_helper_function, params)
