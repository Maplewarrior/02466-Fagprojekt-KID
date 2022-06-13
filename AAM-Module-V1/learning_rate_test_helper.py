
def learning_rate_test_helper_function(input):
    from AAM import AA
    import numpy as np
    import pandas as pd
    
    AAM = AA()
    lr = input[0]
    AA_type = input[1]
    dataframe_size = input[2]
    reps = 10

    AAM.create_synthetic_data(N = dataframe_size, M=21,K=5,sigma=-20,a_param=1,b_param=1000, mute=True)

    losses = []
    for i in range(10):
        AAM.analyse(AA_type = AA_type, mute=True, with_synthetic_data=True,K=5,lr=lr, n_iter=25000, early_stopping=True, with_hot_start=False)
        if AA_type in ["CAA","TSAA"]:
            losses.append(AAM._synthetic_results[AA_type][0].RSS[-1])
        else:
            losses.append(AAM._synthetic_results[AA_type][0].loss[-1])
    
    dataframe = pd.DataFrame.from_dict({
        "TYPE": AA_type,
        "LR": lr,
        "Dataframe Size": dataframe_size,
        "Losses": losses
    })

    print("SAVING")
    csv_name = 'LR results/' + str(AA_type) + "_" + str(lr) + "_" + str(dataframe_size) + ".csv"
    dataframe.to_csv(csv_name, index=False) 
    print("SAVING2")