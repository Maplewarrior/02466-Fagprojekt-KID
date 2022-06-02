import numpy as np
from scipy.special import softmax
from scipy.stats import norm




import numpy as np
from scipy.special import softmax
from scipy.stats import norm
    
    

class _synthetic_data:

    def __init__(self, N, M ,K, p, sigma):
        self.X, self.Z, self.A = self.X(N=N, M=M, K=K, p=p, sigma=sigma)
        self.N = N
        self.M = M
        self.K = K
        self.p = p
        self.columns = ["SQ"+str(i) for i in range(1, M+1)]

    def get_Z(self, M, K, p):
    # Assure reproducibility
        np.random.seed(42)
        betas = np.arange(1,p)*0.5
        
        alphas = np.empty(len(betas)+1)
        Z = np.empty((M, K))
        
        # Calculate beta-values
        betas = softmax(betas)
        
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

    def get_A(self, N, K):
        np.random.seed(42) # set another seed :)
        
        alpha = np.array([1]*K)
        
        return np.random.dirichlet(alpha, size=N).transpose()


    def get_D(self, X_rec, betas, sigma):
        M, N = X_rec.shape
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
                D[j] = (betas[j-1] - X_rec)/(sigma+1e-10) ## Add softplus(sigma)
        
        return D

    def Probs(self, D):
        
        J, M, N = D.shape
        
        probs = np.empty((J-1, M, N)) 
        for i in range(J):
            if i != J-1:
                probs[i,:,:] = norm.cdf(D[i+1], loc=0, scale=1) - norm.cdf(D[i], loc=0, scale=1)
                
        return probs

    def toCategorical(self, probs):
        
        categories = np.arange(1, len(probs)+1)    
        J, M, N = probs.shape
        X_cat = np.empty((M,N))
        
        for m in range(M):
            for n in range(N):
                X_cat[m,n] = np.random.choice(categories, p = list(probs[:,m,n]))
        
        return X_cat
    
    # function combining the previous methods to get X_thilde
    def X(self, M, N, K, p, sigma=1):
        Z, betas = self.get_Z(M, K, p)
        A = self.get_A(N,K)
        X_rec = Z@A
        
        D = self.get_D(X_rec, betas, sigma)
        probs = self.Probs(D)
        X_thilde = self.toCategorical(probs)

        return X_thilde, Z, A

### distribution of answers and check that the data is reproducible ###
""" answer_dist=[]
l = list(X_syn.flatten())
for i in range(p-1):
    answer_dist.append(l.count(i+1))
print(answer_dist)

def assertIdentical(M1, M2):
    N, M = M1.shape
    for i in range(N):
        for j in range(M):
            if (M1[i][j] != M2[i][j]):
                print("not identical")
                return 0
    print("identical")
    return 1
print(assertIdentical(X_syn, X_syn2))
print(type(X_syn))

print(len(np.unique(X_syn)))
print(len(np.unique(Z_syn)))  """ 
