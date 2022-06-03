from AAM import AA

### Set variables ###

"""
###### Setting the variables ######
      
     N: Number of respodents
     M: Number of questions
     K: Number of archetypes
     p: Length of Likert scale
     
     sigma: leakage into other categories (uncertainty in answers)
         - The higher sigma is the harder it will be to find ground truth
         
     a_param: Parameter given to dirichlet distribution to get ground truth A
        
     b_param: Parameter given to dirichlet distribution to get ground truth Z
         - The higher b_param is, the lower the response bias. 
    
"""
N = 50
M = 10
K = 3
p = 5

sigma = 1
a_param = 1
b_param = 1000
rb = True

n_iter = 2000


#%%
AAM = AA()

AA_types = ["RBOAA", "CAA", "OAA",  "TSAA"]
archetypes = [3, 5, 7]

AA_types1 = ["CAA","TSAA"]
# sigma_vals = [-1000000, -10, 0, 0.5, 1, 2, 10, 100, 1000, 10000]
sigma_vals = [-1000, 1, 10]

b_param_vals = []
a_param_vals = []


for type in AA_types:
    for s in sigma_vals:
        AAM.create_synthetic_data(N=N, M=M, K=K, p=p, sigma=s, rb=rb, a_param=a_param, b_param=b_param)
    
        for k in archetypes:
            AAM.analyse(AA_type = type, with_synthetic_data = True, K=k, n_iter = n_iter)
            AAM.save_analysis(filename =  "_sigma_" + str(s) + "_Arche_" + str(k), model_type = type, result_number = 0, with_synthetic_data=True) 
        
            
        