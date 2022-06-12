########## IMPORTS ##########
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from AA_result_class import _OAA_result
from loading_bar_class import _loading_bar
from CAA_class import _CAA
import matplotlib.pyplot as plt 



########## ORDINAL ARCHETYPAL ANALYSIS CLASS ##########
class _OAA:

    loss = []

    def _apply_constraints_AB(self,A):
        m = nn.Softmax(dim=0)
        return m(A)

    def _apply_constraints_beta(self,b): 
        m = nn.Softmax(dim=0)
        return torch.cumsum(m(b), dim=0)[:len(b)-1]

    def _apply_constraints_sigma(self,sigma):
        m = nn.Softplus()
        return m(sigma)

    def _calculate_alpha(self,b):
        b_j = torch.cat((torch.tensor([0.0]),b),0)
        b_j_plus1 = torch.cat((b,torch.tensor([1.0])),0)
        alphas = (b_j_plus1+b_j)/2
        return alphas

    def _calculate_X_tilde(self,X,alphas):
        X_tilde = alphas[X.long()-1]
        return X_tilde

    def _calculate_X_hat(self,X_tilde,A,B):
        # X_hat = B@A@X_tilde.T
        X_hat = X_tilde @ B @ A
        return X_hat

    # def _calculate_D(self,b,X_hat,sigma):
        
    #     D = torch.rand(len(b)+2,len(X_hat),len(X_hat[0,:]))
    #     D[0] = torch.tensor(np.matrix(np.ones((self.M,self.N)) * (-np.inf)))
    #     D[-1] = torch.tensor(np.matrix(np.ones((self.M,self.N)) * (np.inf)))
    #     D[1:-1] = torch.div(b.expand(self.N,self.M,len(b)).T-X_hat,sigma+1e-16)

    #     return D

    def _calculate_loss(self,X, X_hat, b, sigma):

        # stand_norm = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        # D_cdf = stand_norm.cdf(D)
        # P = D_cdf[1:]-D_cdf[:len(D)-1]
        # inverse_log_P = -torch.log(P+1e-16)
        # loss = torch.sum(inverse_log_P[torch.flatten(X.long())-1,self.N_arange,self.M_arange])
        
        # print("-log_P shape: ", inverse_log_P.shape)
        
        pad = nn.ConstantPad1d(1, 0)
        b = pad(b)
        b[-1] = 1.0
        zetaNext = (b[X] - X_hat)/sigma
        zetaPrev = (b[X-1] - X_hat)/sigma
        zetaNext[X==len(b)+1] = float("Inf")
        zetaPrev[X == 1] = -float("Inf")

        #Do phi(zeta1)-phi(zeta2)
        logP= -torch.log((torch.distributions.normal.Normal(0, 1).cdf(zetaNext)- torch.distributions.normal.Normal(0, 1).cdf(zetaPrev))+10E-10) #add small number to avoid underflow.

        loss = torch.sum(logP)
        
        #Find zeta values
        # zetaNext = (torch.cat((torch.tensor([0.0]),b,torch.tensor([1.0])))[Xt.T] - X_hat)/sigma
        # zetaPrev = (torch.cat((torch.tensor([0.0]),b,torch.tensor([1.0])))[Xt.T-1] - X_hat)/sigma
        # zetaNext[Xt.T==len(b)+1] = float("Inf")
        # zetaPrev[Xt.T == 1] = -float("Inf")
        # logP= -torch.log((torch.distributions.normal.Normal(0, 1).cdf(zetaNext)- torch.distributions.normal.Normal(0, 1).cdf(zetaPrev))+10E-10) #add small number to avoid underflow.
        # loss = torch.sum(logP)

        return loss

    def _error(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint):
        
        # print("A shape before:", A_non_constraint.shape)
        
        A = self._apply_constraints_AB(A_non_constraint)
        # print("A shape after", A.shape)
        
        # print("B before", B_non_constraint.shape)
        B = self._apply_constraints_AB(B_non_constraint)
        # print("B shape after", B.shape)
        b = self._apply_constraints_beta(b_non_constraint)
        sigma = self._apply_constraints_sigma(sigma_non_constraint)
        alphas = self._calculate_alpha(b)
        
        # print("X_shape", Xt.shape)
        X_tilde = self._calculate_X_tilde(Xt,alphas)
        # print("X_tilde shape", X_tilde.shape)
        X_hat = self._calculate_X_hat(X_tilde,A,B)
        # print("X_hat shape", X_hat.shape)
        # D = self._calculate_D(b,X_hat,sigma)
        # print("D shape", D.shape)
        loss = self._calculate_loss(Xt, X_hat, b, sigma)

        return loss
        

    def _compute_archetypes(self, X, K, p, n_iter, lr, mute, columns, with_synthetic_data = False, early_stopping = False, with_CAA_initialization: bool = False):

        
        ########## INITIALIZATION ##########
        self.M = len(X)
        self.N = len(X[0,:])
        self.N_arange = [n for n in range(self.M) for m in range(self.N)]
        self.M_arange = [m for m in range(self.N) for n in range(self.M)]
        self.loss = []
        start = timer()
        N, _ = X.T.shape

        # if with_CAA_initialization:
        #     CAA = _CAA()
        #     initialization_result = CAA._compute_archetypes(X, K, n_iter, lr, True,columns,with_synthetic_data = False, early_stopping = True)
        #     A_non_constraint = torch.autograd.Variable(torch.tensor(initialization_result.A), requires_grad=True)
        #     B_non_constraint = torch.autograd.Variable(torch.tensor(initialization_result.B), requires_grad=True)
            
    
        A_non_constraint = torch.autograd.Variable(torch.randn(K, N), requires_grad=True)
        # B_non_constraint = torch.autograd.Variable(torch.randn(N, K), requires_grad=True) ### OLD ###
        
        
        B_non_constraint=torch.sparse_csr_tensor(torch.tensor(range(self.N+1)),torch.tensor(np.random.randint(0, K,self.N, dtype=np.int64)),torch.ones(self.N),(self.N,K)).to_dense()
        B_non_constraint = B_non_constraint*np.log(N*2)
        B_non_constraint.requires_grad=True
        
        
        Xt = torch.tensor(X, dtype = torch.long)
        b_non_constraint = torch.autograd.Variable(torch.rand(p), requires_grad=True)
        sigma_non_constraint = torch.autograd.Variable(torch.rand(1), requires_grad=True)
        optimizer = optim.Adam([A_non_constraint, 
                                B_non_constraint, 
                                b_non_constraint, 
                                sigma_non_constraint], amsgrad = True, lr = lr)

        if not mute:
            loading_bar = _loading_bar(n_iter, "Ordinal Archetypal Analysis")

        

        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint)
            self.loss.append(L.detach().numpy())
            L.backward()
            optimizer.step()


            ########## EARLY STOPPING ##########
            if i % 25 == 0 and early_stopping:
                if len(self.loss) > 200 and (self.loss[-round(len(self.loss)/100)]-self.loss[-1]) < ((self.loss[0]-self.loss[-1])*1e-4):
                    if not mute:
                        loading_bar._kill()
                        print("Analysis ended due to early stopping.\n")
                    break
            
        
        ########## POST ANALYSIS ##########
        #Z_f = (X@self._apply_constraints_AB(B_non_constraint).detach().numpy())
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint)
        alphas_f = self._calculate_alpha(b_f)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (X_tilde_f@self._apply_constraints_AB(B_non_constraint).detach().numpy())
        X_hat_f = self._calculate_X_hat(X_tilde_f,A_f,B_f)
        end = timer()
        time = round(end-start,2)
        Z_f = (A_f@X_tilde_f.T).T
        
        # print(X_hat_f)

        # print(b_f)
        # print(sigma_non_constraint)

        result = _OAA_result(A_f,B_f,X,n_iter,b_f.detach().numpy(),Z_f,X_tilde_f,Z_tilde_f,X_hat_f,self.loss,K,time,columns,"OAA",with_synthetic_data=with_synthetic_data)

        if not mute:
            result._print()
        
        return result