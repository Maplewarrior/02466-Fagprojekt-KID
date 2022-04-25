import numpy as np
from scipy.special import softmax
from scipy.stats import norm


def betaConstraintsNoBias(betas):
    
    new_betas = np.empty(len(betas))
    denom = sum(betas)
    
    for i in range(len(new_betas)):
        new_betas[i] = np.sum(betas[:i]) / denom
    
    # betas = np.concatenate((np.array([betas[0]]), betas))
    # betas = softmax(betas)
    # new_betas = np.cumsum(betas, axis = 0)[:len(betas)-1]
    
    return new_betas


def softPlus(sigma):
    return np.log(1+np.exp(sigma))

def get_Z(M, K, p):
    # Assure reproducibility
    np.random.seed(123)
    
    # betas = np.arange(1,p)
    # betas = np.ones(p-1)*5
    betas = np.arange(1,p)
    betas = betaConstraintsNoBias(betas)
    
    betas = np.array([0.00, 0.4, 0.8, 1.2, 1.6])
    
    alphas = np.empty(len(betas)+1)
    
    # Calculate alpha-values
    alphas[0] = (0 + betas[0]) / 2
    alphas[-1] = (1+ betas[-1]) / 2
    for i in range(len(betas)-1):
        alphas[i+1] = (betas[i] + betas[i+1]) / 2
        

    
    Z = np.empty((M, K))
    # Draw samples from the alphas to construct Z
    
    for i in range(M):
        for j in range(K):
            idx = np.random.randint(0, len(alphas))
            # Z[i,j] = np.random.choice(alphas, size=1)
            Z[i,j] = alphas[idx]
    
    
    # # "non-random Z"
    # cutoff_vals = [int(np.round( ((i+1) * len(alphas) - 1) / 4)) for i in range(4)]
    # # print(cutoff_vals)
    # # print(cutoff_vals)
    # # print(alphas)
    # p_deviation = 0.4
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
    np.random.seed(123) # set another seed :)
    
    a = np.array([1]*K)
    return np.random.dirichlet(a, size=N).transpose()


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
            D[j] = (betas[j-1] - X)/(softPlus(sigma)) # Softplus = np.log(1+np.exp(sigma))

    return D

def Probs(D):
    
    J, M, N = D.shape
    
    probs = np.empty((J-1, M, N)) 
    
    # D_cdf = stand_norm.cdf(D)
    # P = D_cdf[1:]-D_cdf[:len(D)-1]
    
    probs = norm.cdf(D[1:], loc=0, scale=1)-norm.cdf(D[:len(D)-1], loc=0, scale=1)
    
    # for i in range(J):
    #     if i != J-1:
    #         probs[i,:,:] = norm.cdf(D[i+1], loc=0, scale=1) - norm.cdf(D[i], loc=0, scale=1)
    
    
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

M = 15
N = 1000
p = 6
K = 5
sigma = 1


Z, betas = get_Z(M,K,p)
A = get_A(N,K)
X_raw = Z@A
D = get_D(X_raw, betas, sigma)

probs = Probs(D)
X_cat = toCategorical(probs)

J, _,_ = D.shape

# print("betas", betas)
# # print("rå data", X_raw)
# print("mean:", np.mean(X_raw))
# print("median:", np.median(X_raw))

print("betas: ", betas)


# for i in range(J):
#     print(D[i,0,0])
# print("probs:", (probs[0,0,0]))
# for i in range(J):
#     print(D[i,1,1])
# print("probs:", (probs[0,1,1]))
# for i in range(J):
#     print(D[i,2,2])
# print("probs:", (probs[0,2,2]))
# for i in range(J):
#     print(D[i,3,3])

# print(probs[0,3,3])




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

#%%

