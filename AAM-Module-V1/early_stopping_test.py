
if __name__ ==  '__main__':
    import numpy as np
    from early_stopping_helper_function import early_stopping_helper_function
    import multiprocessing

    AA_types = ["CAA","RBOAA","OAA"]
    early_stoppings = [False,True]
    reps = [1,2,3,4,5,6,7,8,9,10]
    params = []

    for AA_type in AA_types:
        for early_stopping in early_stoppings:
            for rep in reps:
                params.append([AA_type,early_stopping,rep])

    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        p.map(early_stopping_helper_function, params)