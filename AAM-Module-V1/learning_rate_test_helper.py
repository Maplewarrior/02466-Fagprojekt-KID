
def learning_rate_test_helper_function(input):
    from AAM import AA
    import numpy as np
    
    AAM = AA()
    lr = input[0]
    AA_type = input[1]
    reps = 10
    
    AAM.create_synthetic_data(N = 5000, M=21,K=5,sigma=-20,a_param=1,b_param=1000, mute=True)

    losses = []
    for i in range(10):
        AAM.analyse(AA_type = AA_type, mute=True, with_synthetic_data=True,K=5,lr=lr, n_iter=25000, early_stopping=True)
        if AA_type in ["CAA","TSAA"]:
            losses.append(AAM._synthetic_results[AA_type][0].RSS[-1])
        else:
            losses.append(AAM._synthetic_results[AA_type][0].loss[-1])
    
    print("\n\nTYPE: " + str(AA_type) + " LR: " + str(lr) + " LOSSES: " + str(losses) + "\n\n")
