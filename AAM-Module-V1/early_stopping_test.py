
if __name__ ==  '__main__':
    import numpy as np
    from early_stopping_helper_function import early_stopping_helper_function
    import multiprocessing

    AA_types = ["CAA","RBOAA","OAA"]
    early_stoppings = [False,True]
    params = []

    for AA_type in AA_types:
        for early_stopping in early_stoppings:
            params.append([AA_type,early_stopping])

    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(early_stopping_helper_function, params)