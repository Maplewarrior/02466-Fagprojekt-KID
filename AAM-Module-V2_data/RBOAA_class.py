########## IMPORTS ##########
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from AA_result_class import _OAA_result
from loading_bar_class import _loading_bar



########## ORDINAL ARCHETYPAL ANALYSIS CLASS ##########
class _RBOAA:

    loss = []

    def _apply_constraints_AB(self,A):
        m = nn.Softmax(dim=0)
        return m(A)

    def _apply_constraints_beta(self,b,J):    
        m = nn.Softmax(dim=1)
        return torch.cumsum(m(b), dim=1)[:,:J-1]

    def _apply_constraints_sigma(self,sigma):
        m = nn.Softplus()
        return m(sigma)

    def _calculate_alpha(self,b,J):

        zeros = torch.zeros((len(b),1))
        ones = torch.ones((len(b),1))

        b_j = torch.cat((zeros,b),1)
        b_j_plus1 = torch.cat((b,ones),1)

        alphas = (b_j_plus1+b_j)/2

        return alphas

    def _calculate_X_tilde(self,X,alphas):
        N = len(X)
        M = len(X[0,:])
        N_arange = [n for n in range(N) for m in range(M)]
        X_tilde = torch.reshape(alphas[N_arange,torch.flatten(X-1)],(X.shape))
        return X_tilde
        
    def _calculate_X_hat(self,X_tilde,A,B):
        return X_tilde@B@A

    def _calculate_D(self,b,Xt,X_hat,sigma):

        N = len(Xt[0,:])
        M = len(Xt)
        J = len(b[0,:])
        D = torch.rand(J+2,N,M)

        for j in range(J+2):
            if j == 0:
                D[j] = torch.tensor(np.matrix(np.ones((M)) * (-np.inf)))
            elif j == J+1:
                D[j] = torch.tensor(np.matrix(np.ones((M)) * (np.inf)))
            else:
                D[j] = torch.div((b[:,j-1] - X_hat[:, None]),sigma)[:,0,:].T

        return D

    def _calculate_loss(self,D,X):
        
        N = len(X[0,:])
        M = len(X)
        stand_norm = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        D_cdf = stand_norm.cdf(D)
        P = D_cdf[1:]-D_cdf[:len(D)-1]
        inverse_log_P = -torch.log(P)

        N_arange = [n for n in range(N) for m in range(M)]
        M_arange = [m for m in range(M) for n in range(N)]

        loss = torch.sum(inverse_log_P[torch.flatten(X)-1,N_arange,M_arange])

        return loss

    def _error(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,J):
        
        A = self._apply_constraints_AB(A_non_constraint)
        B = self._apply_constraints_AB(B_non_constraint)
        b = self._apply_constraints_beta(b_non_constraint,J)
        sigma = self._apply_constraints_sigma(sigma_non_constraint)
        alphas = self._calculate_alpha(b,J)
        
        X_tilde = self._calculate_X_tilde(Xt,alphas)
        X_hat = self._calculate_X_hat(X_tilde,A,B)
        D = self._calculate_D(b,Xt,X_hat,sigma)
        loss = self._calculate_loss(D,Xt)

        return loss
        
    def _compute_archetypes(self, X, K, n_iter, lr, mute,columns,with_synthetic_data = False):

        ########## INITIALIZATION ##########
        self.loss = []
        start = timer()
        if not mute:
            loading_bar = _loading_bar(n_iter, "Response Bias Ordinal Arhcetypal Analysis")
        N, _ = X.T.shape
        J = int((np.max(X)-np.min(X))+1)
        Xt = torch.autograd.Variable(torch.tensor(X), requires_grad=False)
        A_non_constraint = torch.autograd.Variable(torch.rand(K, N), requires_grad=True)
        B_non_constraint = torch.autograd.Variable(torch.rand(N, K), requires_grad=True)
        b_non_constraint = torch.autograd.Variable(torch.rand(N,J), requires_grad=True)
        sigma_non_constraint = torch.autograd.Variable(torch.rand(N), requires_grad=True)
        optimizer = optim.Adam([A_non_constraint, 
                                B_non_constraint, 
                                b_non_constraint, 
                                sigma_non_constraint], amsgrad = True, lr = lr)
        

        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,J)
            self.loss.append(L)
            L.backward()
            optimizer.step() 
        
        
        ########## POST ANALYSIS ##########
        loss_f = self._error(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,J).item()
        Z_f = (X@self._apply_constraints_AB(B_non_constraint).detach().numpy())
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint,J)
        alphas_f = self._calculate_alpha(b_f,J)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (X_tilde_f@self._apply_constraints_AB(B_non_constraint).detach().numpy())
        X_hat_f = self._calculate_X_hat(X_tilde_f,A_f,B_f)
        end = timer()
        time = round(end-start,2)
        result = _OAA_result(A_f,B_f,X,n_iter,b_f,Z_f,X_tilde_f,Z_tilde_f,X_hat_f,self.loss,K,time,columns,"RBOAA",with_synthetic_data=with_synthetic_data)

        if not mute:
            result._print()
        
        return result