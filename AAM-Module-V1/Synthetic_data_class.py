import numpy as np

class syntheticData:
    
    def Z(self, M, K, p):
        # Assure reproducibility
        np.random.seed(42)
        
        betas = np.arange(1, p)
        alphas = np.empty(len(betas))
        Z = np.empty((M, K))
        
        # Calculate beta-values
        betas = betas / sum(betas)
        
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
        
        return Z, betas
    
    
    def A(self, N, K):
        np.random.seed(42) # set another seed :)
        
        alpha = np.ones(K)
        
        return np.random.dirichlet(alpha, size=N).transpose()
    
    def map_X_noise_free(self, X, betas):
        
        M, N = X.shape
        X_thilde = np.empty((M,N))
        
        for i in range(M):
            for j in range(N):
                for k in range(len(betas)):
                    if not k == len(betas)-1:
                        if betas[k] <= X[i,j] and X[i,j] <= betas[k+1]:
                            X_thilde[i,j] = int(k+1)
                            
                        elif X[i,j] < betas[0]:
                            X_thilde[i,j] = int(1)
                        
                        elif X[i,j] > betas[-1]:
                            X_thilde[i,j] = int(len(betas))
        X_thilde = X_thilde.astype(int)
        return X_thilde    
    
    def X(self, N, M, K, p, noise=False):
        
        Z, betas = self.Z(M=M, K=K, p=p)
        A = self.A(N=N, K=K)
        X_hat = Z@A
        
        X_thilde = self.map_X_noise_free(X=X_hat, betas=betas)
        
        return X_thilde, Z, A
    
    
    
#### Testing the script ####   
## Define constants
""" N = 1000 # number of respondents
M = 10 # number of questions
K = 5 # number of archetypes
p = 10 # length of likert scale

syn = syntheticData()
X_syn, Z_syn, A_syn = syn.X(N=N, M=M, K=K, p=p)



syn2 = syntheticData()
X_syn2, _, _ = syn2.X(N=N, M=M, K=K, p=p)


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
print(len(np.unique(Z_syn))) """




    
    