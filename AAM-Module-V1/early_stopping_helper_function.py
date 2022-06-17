

def early_stopping_helper_function(params):
    from AAM import AA
    import numpy as np
    import pandas as pd

    AAM = AA()
    
    AA_type = params[0]
    early_stopping = params[1]
    reps = 30

    losses = []

    AAM.create_synthetic_data(10000,21,5,6,-20,False,1,1,True,0)

    for rep in range(reps):
        print(rep)
        AAM.analyse(5,6,2000,early_stopping,AA_type,0.01,True,True,True)
        losses.append(AAM._synthetic_results[AA_type][0].loss[0])

    dataframe = pd.DataFrame.from_dict({
            'AA type': AA_type,
            'early_stopping': early_stopping,
            'losses': losses})
    
    csv_name = "early stopping results/ES_{0}_K={1}".format(AA_type,early_stopping)
    dataframe.to_csv(csv_name, index=False) 
    



    

