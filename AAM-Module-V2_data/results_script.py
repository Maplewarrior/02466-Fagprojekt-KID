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

N = 10000
M = 10
K = 4
p = 5
sigma = 1
a_param = 0.05
b_param = 10000

n_iter = 2000


#%%


AAM = AA()

# Make sure this includes a_param and b_param
AAM.create_synthetic_data(N=N, M=M, K=K, p=p, sigma=sigma)

# AAM.analyse(AA_type = "CAA", with_synthetic_data = True,K=K)

#%%
AA_types = ["CAA", "OAA", "ROBAA", "TSAA"]
archetypes = [3, 5, 7]


# sigma_vals = [-1000000, -10, 0, 0.5, 1, 2, 10, 100, 1000, 10000]
sigma_vals = [-10000, 1, 10]

b_param_vals = []
a_param_vals = []

for s in sigma_vals:
    AAM.create_synthetic_data(N=N, M=M, K=K, p=p, sigma=s)
    for k in archetypes:
        AAM.analyse(AA_type = "CAA", with_synthetic_data = True, K=k)
        
    