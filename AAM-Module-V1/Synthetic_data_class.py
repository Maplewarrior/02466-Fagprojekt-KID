import numpy as np
from scipy.special import softmax
import scipy.stats

class syntheticData:
    
    """ def applySoftmax(M):
        return softmax(M, axis=0) """

    def Z(self, M, K, p):
        # Assure reproducibility
        np.random.seed(42)
        
        betas = np.arange(1, p)
        alphas = np.empty(len(betas))
        Z = np.empty((M, K))
        
        # Calculate beta-values
        betas = betas/sum(betas)
        
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
        """
        Implement version with noise only..
        """
        M, N = X.shape
        X_thilde = np.empty((M,N))
        
        for i in range(M):
            for j in range(N):
                for k in range(len(betas)):
                    if not k == len(betas)-1: #if not done
                        if betas[k] <= X[i,j] and X[i,j] <= betas[k+1]:
                            X_thilde[i,j] = int(k+1)
                            
                        elif X[i,j] < betas[0]:
                            X_thilde[i,j] = int(1)
                        
                        elif X[i,j] > betas[-1]:
                            X_thilde[i,j] = int(len(betas))
        X_thilde = X_thilde.astype(int)
        return X_thilde    
    
    def map_X(self, X, betas, sigma):
        M, N = X.shape
        X_thilde = np.empty((M,N))
        D = np.empty((M,N,len(betas)+2))
        probs = np.empty((M,N,len(betas)))

        for i in range(M):
            for j in range(N):
                for k in range(len(betas)):
                    if not k == len(betas)-1:
                        D[i,j,k] = (betas[k] - X[i,j]) / sigma

                        probs[i,j,k] = scipy.stats.norm.cdf(D[i,j,k+1]) - scipy.stats.norm.cdf(D[i,j,k])
        print("D matrix ",D.shape)
        print(D[0:5,0,:])
        print("probs",probs.shape)
        print(probs[0:5,0,:])
        print("cumsum ",np.cumsum(probs[0:5,0,:]))
        # nu tages en værdi mellem 0 og 1 og mapper til en kategori
        
        #np.random.rand(0,1)
        #X_thilde[i,j] = int(k+1)

        X_thilde = X_thilde.astype(int)
        return X_thilde    

    def X(self, N, M, K, p, sigma):
        
        Z, betas = self.Z(M=M, K=K, p=p)
        A = self.A(N=N, K=K)
        X_hat = Z@A
        
        X_thilde = self.map_X(X=X_hat, betas=betas, sigma=sigma)
        
        return X_thilde, Z, A
    
    
    
#### Testing the script ####   
## Define constants
N = 10 # number of respondents
M = 10 # number of questions
K = 5 # number of archetypes
p = 10 # length of likert scale
sigma = 0.3

syn = syntheticData()
X_syn, Z_syn, A_syn = syn.X(N=N, M=M, K=K, p=p, sigma=sigma) 

answer_dist=[]
l = list(X_syn.flatten())
for i in range(p-1):
    answer_dist.append(l.count(i+1))
print(answer_dist)


"""

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
print(len(np.unique(Z_syn))) 
"""