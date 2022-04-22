import numpy as np
from scipy.special import softmax
from scipy.stats import norm


def betaConstraints(betas):
    
    betas = betas / sum(betas)
    new_betas = np.empty(len(betas))
    
    for i in range(len(betas)):
        new_betas[i] = sum(betas[:i])
   
    return new_betas


def softPlus(sigma):
    return np.log(1+np.exp(sigma))

def get_Z(M, K, p):
    # Assure reproducibility
    np.random.seed(42)
    
    # betas = np.arange(1,p)
    betas = np.ones(p-1)*5
    # Calculate beta-values
    betas = betaConstraints(betas)
    # betas = softmax(betas)
    
    
    alphas = np.empty(len(betas)+1)
    Z = np.empty((M, K))
    
   
    
    # Calculate alpha-values
    for i in range(len(betas)):
        if not i == (len(betas)-1) or 0:
            alphas[i] = (betas[i]+betas[i+1])/2
        elif i == 0:
            alphas[i] = (0+betas[i])/2
        elif i == (len(betas)-1):
            alphas[i] = (betas[i]+1)/2
    
    # Draw samples from the alphas to construct Z
    
    for i in range(M):
        for j in range(K):
            Z[i,j] = np.random.choice(alphas, size=1)
    
    
    # "non-random Z"
    # cutoff_vals = [int(np.round( ((i+1) * len(alphas) - 1) / 4)) for i in range(4)]
    # # print(cutoff_vals)
    # # print(cutoff_vals)
    # # print(alphas)
    # p_deviation = 0
    # for i in range(M):
    #     for j in range(K):
    #         if not np.random.choice([0,1], p=(1-p_deviation, p_deviation)) == 1:
    #             if j <= np.round(K/3):
    #                 Z[i,j] = np.random.choice(alphas[:cutoff_vals[0]], size=1)
    #             elif j <= np.round(2*K/3):
    #                 Z[i,j] = np.random.choice(alphas[cutoff_vals[0]:cutoff_vals[1]], size=1)
    #             else:
    #                 Z[i,j] = np.random.choice(alphas[cutoff_vals[1]:cutoff_vals[2]], size=1)
    #         else:
    #             Z[i,j] = np.random.choice(alphas, size=1)
    # print("n.o. alphas: ", len(alphas))
    return Z, betas

def get_A(N, K):
    np.random.seed(42) # set another seed :)
    
    alpha = np.array([1]*K)
    
    return np.random.dirichlet(alpha, size=N).transpose()


def get_D(X, betas, sigma):
    M, N = X.shape
    J = len(betas)
    
    X_thilde = np.empty((M,N))
    D = np.empty((J+2, M, N))
    
    for j in range(J+2):
        # Left-most tail
        if j == 0:
            D[j] = np.ones((M,N))*(np.inf*(-1))
        # Right-most tail
        elif j == J+1:
            D[j] = np.ones((M,N))*(np.inf)
        else:
            D[j] = (betas[j-1] - X)/(softPlus(sigma)) ## Add softplus(sigma)
    return D

def Probs(D):
    
    J, M, N = D.shape
    
    probs = np.empty((J-1, M, N)) 
    for i in range(J):
        if i != J-1:
            probs[i,:,:] = norm.cdf(D[i+1], loc=0, scale=1) - norm.cdf(D[i], loc=0, scale=1)
            
    return probs

def toCategorical(probs):
    
    categories = np.arange(1, len(probs)+1)    
    J, M, N = probs.shape
    X_cat = np.empty((M,N))
    
    for m in range(M):
        for n in range(N):
            X_cat[m,n] = np.random.choice(categories, p = list(probs[:,m,n]))
    
    return X_cat
    
    
#%%

M = 20
N = 10000
p = 5
K = 3
sigma = 0.001


Z, betas = get_Z(M,K,p)
A = get_A(N,K)
X_raw = Z@A
D = get_D(X_raw, betas, sigma)

probs = Probs(D)
X_cat = toCategorical(probs)

#%%
print("betas", betas)
# print(X_raw)


# print("---------------------------------------")
# print(D)


print("X_categorical's distribution of answers:")
a_dist = []
for i in range(p):
    a_dist.append(len(X_cat.flatten()[X_cat.flatten()==i+1]))
print(a_dist)

#%%
# print(probs)

answer_dist = []
for p in range(len(probs)):
    answer_dist.append(np.count_nonzero(probs[p,:,:].flatten()))
    
print(answer_dist)
print(sum(answer_dist))
#%%
# print(Z)
# print("beta: ", betas)
print(np.round(probs,2))
print(probs.shape)



#%%
s_probs = 0
for i in range(len(probs)):
    s_probs += probs[i]
print(s_probs)


