

def ESS8_Final_Result_helper_function(params):
    from AAM import AA
    import numpy as np

    AAM = AA()
    
    AA_type = params[0]
    K = params[1]

    AAM.load_csv("ESS8_data.csv",np.arange(12,33))
    AAM.analyse(K,6,10000,True,AA_type,0.01,True,False,True)
    AAM.save_analysis("K={0} - {1}".format(K,AA_type),AA_type)

