import numpy as np
from scipy.stats import norm

class _synthetic_data:
    
    def __init__(self, N, M ,K, p, sigma, rb, a_param, b_param):
        
        self.X, self.Z, self.A = self.X(N=N, M=M, K=K, p=p, sigma=sigma, rb=rb, a_param=a_param, b_param=b_param)
        self.N = N
        self.M = M
        self.K = K
        self.p = p
        self.columns = ["SQ"+str(i) for i in range(1, M+1)]
        
    # If there's response bias, sample from a dirichlet distribution.
    def biasedBetas(self, N, p, b_param):
        b = np.array([b_param]*p)
        return np.random.dirichlet(b, size=N)
    
    def betaConstraintsBias(self, betas):
        N, J = betas.shape
        new_betas = np.empty((N,J))
    
        denoms = np.sum(betas,axis=1)
        
        for i in range(N):
            for j in range(J):
                new_betas[i,j] = np.sum(betas[i,:j+1])/denoms[i]
    
        # Return and remove the column of ones
        return new_betas[:,:-1]
    
    def betaConstraints(self, betas):
   
       new_betas = np.empty(len(betas))
       denom = sum(betas)
       
       for i in range(len(new_betas)):
           new_betas[i] = np.sum(betas[:i+1]) / denom
   
       return new_betas[:-1]
   
    def softplus(self, sigma):
        return np.log(1 + np.exp(sigma))

    # rb = response bias
    def get_Z(self, N, M, K, p, rb, b_param):
        # Ensure reproducibility
        np.random.seed(42)
        
        Z = np.empty((M,K))
        
        
        # Check if we want to model response bias
        if not rb:
            betas = np.ones(p)
            betas = self.betaConstraints(betas)
            alphas = np.empty(p)
            alphas[0] = (0 + betas[0]) / 2
            alphas[-1] = (1+ betas[-1]) / 2
            for i in range(len(betas)-1):
                alphas[i+1] = (betas[i] + betas[i+1]) / 2
            
            for i in range(M):
                for j in range(K):
                    Z[i,j] = np.random.choice(alphas, size=1)
            
        else:
            betas = self.biasedBetas(N=N, p=p, b_param=b_param)
            betas = self.betaConstraintsBias(betas)
            alphas = np.empty((N,p))
            
            for i in range(N):
                # Set start and end values
                alphas[i,0] = (0+betas[i,0])/2
                alphas[i,-1] = (1 + betas[i,-1])/2
                for j in range(p-2):
                    alphas[i,j+1] = (betas[i,j]+betas[i,j+1]) / 2
            
            # Get Z matrix
            Z = np.empty((M,K))
            for m in range(M):
                for l in range(K):
                # for k in range(N):
                    row_idx = np.random.randint(0, N)
                    col_idx = np.random.randint(0, p)
                                   
                    Z[m,l] = alphas[row_idx, col_idx]
        return Z, betas

    def get_A(self, N, K, a_param):
        np.random.seed(42) # set another seed :)
        
        alpha = np.array([a_param]*K)
        return np.random.dirichlet(alpha, size=N).transpose()
    
    def get_D(self, X_rec, betas, sigma, rb):
        
        M, N = X_rec.shape
        
        
        if rb == False:
        
            J = len(betas)    
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
                    
        else:
            J = len(betas[0,:])
            D = np.empty((J+2, M, N))
            
            for j in range(J+2):
                if j == 0:
                    D[j] = np.ones((M,N))*(np.inf*(-1))
                elif j == J+1:
                    D[j] = np.ones((M,N))*(np.inf)
                else:
                    D[j] = (betas[:,j-1] - X_rec)/(sigma+1e-10) ## Add softplus(sigma)
                    # D[j] = torch.div((b[:,j-1] - X_hat[:, None]),sigma)[:,0,:].T
        
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
                X_cat[m,n] = int(np.random.choice(categories, p = list(probs[:,m,n])))
                
        X_cat = X_cat.astype(int)
        return X_cat
    
    # function combining the previous methods to get X_thilde
    def X(self, M, N, K, p, sigma, rb=False, a_param=1, b_param=100):
        
        Z, betas = self.get_Z(N=N,M=M, K=K, p=p, rb=rb, b_param=b_param)
        A = self.get_A(N, K, a_param=a_param)
        X_rec = Z@A
        
        D = self.get_D(X_rec, betas, self.softplus(sigma), rb=rb)
        probs = self.Probs(D)
        X_thilde = self.toCategorical(probs)
        
        
        return X_thilde, Z, A
        
        
            
        
    