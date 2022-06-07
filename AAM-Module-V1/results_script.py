from AAM import AA

### Set variables ###

"""
###### Setting the variables ######
      
     N: Number of respondents
     M: Number of questions
     K: Number of archetypes
     p: Length of Likert scale
     
     sigma: leakage into other categories (uncertainty in answers)
         - The higher sigma is the harder it will be to find ground truth
         
     a_param: Parameter given to dirichlet distribution to get ground truth A
        
     b_param: Parameter given to dirichlet distribution to get ground truth Z
         - The higher b_param is, the lower the response bias. 
    
"""
N = 1000
M = 21
K = 5
p = 6

#sigma = 1
a_param = 1
b_param = 1000
rb = True

n_iter = 3000

import torch
torch.manual_seed(2)

import random
random.seed(2)

#%%
AAM = AA()

#AA_types = ["RBOAA", "CAA", "OAA",  "TSAA"]
AA_types = ["RBOAA"]
#archetypes = [3, 5, 7 ]
archetypes = [5]

#sigma_vals = [-1000000, -10, 0, 0.5, 1, 2, 10, 100, 1000, 10000]
sigma_vals = [-5, -1,-0.75, -0.5, -0.25, 0, 0.5, 1, 5, 10]
#sigma_vals = [1]

b_param_vals = []
a_param_vals = []

#import numpy as np
#for i in sigma_vals:
#    print(np.log(1 + np.exp(i)))


for type in AA_types:
    for s in sigma_vals:
        AAM.create_synthetic_data(N=N, M=M, K=K, p=p, sigma=s, rb=rb, a_param=a_param, b_param=b_param)
    
        for k in archetypes:
            AAM.analyse(AA_type = type, with_synthetic_data = True, K=k, n_iter = n_iter)
            AAM.save_analysis(filename =  "_sigma_" + str(s) + "_Arche_" + str(k), model_type = type, result_number = 0, with_synthetic_data=True) 

        
            

