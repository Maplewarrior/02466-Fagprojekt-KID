########## IMPORTS ##########
from tokenize import Double
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from AA_result_class import _OAA_result
from loading_bar_class import _loading_bar
from OAA_class import _OAA


# IMPORT TO TEST DIVERGENCE OF MODEL
from synthetic_data_class import _synthetic_data

########## ORDINAL ARCHETYPAL ANALYSIS CLASS ##########
class _RBOAA:

    loss = []

    ########## HELPER FUNCTION // A AND B ##########
    def _apply_constraints_AB(self,A):
        m = nn.Softmax(dim=0)
        return m(A)

    ########## HELPER FUNCTION // BETAS ##########
    def _apply_constraints_beta(self,b,J):    
        m = nn.Softmax(dim=1)
        return torch.cumsum(m(b), dim=1)[:,:J-1]

    ########## HELPER FUNCTION // SIGMA ##########
    def _apply_constraints_sigma(self,sigma):
        m = nn.Softplus()
        return m(sigma)

    ########## HELPER FUNCTION // ALPHAS ##########
    def _calculate_alpha(self,b,J):
        b_j = torch.cat((torch.zeros((len(b),1)),b),1)
        b_j_plus1 = torch.cat((b,torch.ones((len(b),1))),1)
        alphas = (b_j_plus1+b_j)/2
        return alphas

    ########## HELPER FUNCTION // X TILDE ##########
    def _calculate_X_tilde(self,X,alphas):
        X_tilde = torch.reshape(alphas[self.N_arange,torch.flatten(X.long()-1)],(X.shape))
        return X_tilde
    
    ########## HELPER FUNCTION // X HAT ##########
    def _calculate_X_hat(self,X_tilde,A,B):
        return X_tilde@B@A

    ########## HELPER FUNCTION // LOSS ##########
    def _calculate_loss(self,X, X_hat, b, sigma):
        
        pad = nn.ConstantPad1d(1, 0)
        b = pad(b)
        b[:,-1] = 1.0
        
        zetaNext = (torch.gather(b,1,X)-X_hat)/sigma
        zetaPrev = (torch.gather(b,1,X-1)-X_hat)/sigma
        zetaNext[X==len(b)+1] = float("Inf")
        zetaPrev[X == 1] = -float("Inf")

        logP= -torch.log((torch.distributions.normal.Normal(0, 1).cdf(zetaNext)- torch.distributions.normal.Normal(0, 1).cdf(zetaPrev))+10E-10) #add small number to avoid underflow.
        loss = torch.sum(logP)

        return loss

    ########## HELPER FUNCTION // ERROR ##########
    def _error(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,J):
        
        A = self._apply_constraints_AB(A_non_constraint)
        B = self._apply_constraints_AB(B_non_constraint)
        b = self._apply_constraints_beta(b_non_constraint,J)
        sigma = self._apply_constraints_sigma(sigma_non_constraint)
        alphas = self._calculate_alpha(b,J)
        
        X_tilde = self._calculate_X_tilde(Xt,alphas)
        X_hat = self._calculate_X_hat(X_tilde,A,B)
        loss = self._calculate_loss(Xt, X_hat, b, sigma)

        return loss
    
    ########## PERFORMING ARCHEYPAL ANALYSIS ##########
    def _compute_archetypes(self, X, K, p, n_iter, lr, mute,columns,with_synthetic_data = False, early_stopping = False, with_OAA_initialization: bool = False):

        ########## INITIALIZATION ##########
        self.N = len(X[0,:])
        self.M = len(X)
        self.N_arange = [n for n in range(self.N) for m in range(self.M)]
        self.M_arange = [m for m in range(self.M) for n in range(self.N)]
        self.loss = []
        start = timer()

        if not mute:
            loading_bar = _loading_bar(n_iter, "Response Bias Ordinal Archetypal Analysis")
        N, _ = X.T.shape
        Xt = torch.tensor(X, dtype=torch.long)

        ########## CAA INITIALIZATION ##########
        if with_OAA_initialization:
            if not mute:
                print("Performing OAA for initialization of ROBAA.")
            OAA = _OAA()
            initialization_result = OAA._compute_archetypes(X, K, p, n_iter, lr, mute, columns, with_synthetic_data = with_synthetic_data, early_stopping = early_stopping)
            A_non_constraint = torch.autograd.Variable(torch.tensor(initialization_result.A), requires_grad=True)
            B_non_constraint = torch.autograd.Variable(torch.tensor(initialization_result.B), requires_grad=True)
            sigma_non_constraint = torch.autograd.Variable(torch.tensor(initialization_result.sigma).repeat_interleave(N), requires_grad=True)
        else:
            A_non_constraint = torch.autograd.Variable(torch.randn(K, N), requires_grad=True)
            B_non_constraint = torch.autograd.Variable(torch.randn(N, K), requires_grad=True)
            sigma_non_constraint = torch.autograd.Variable(torch.rand(N)*(-4), requires_grad=True)

        ########## INITIALIZATION OF GENERAL VARIABLES ##########
        b_non_constraint = torch.autograd.Variable(torch.rand(N,p), requires_grad=True)
        optimizer = optim.Adam([A_non_constraint, 
                                B_non_constraint, 
                                b_non_constraint, 
                                sigma_non_constraint], amsgrad = True, lr = lr)
        

        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint,p)
            self.loss.append(L.detach().numpy())
            L.backward()
            optimizer.step() 

            ########## EARLY STOPPING ##########
            if i % 25 == 0 and early_stopping:
                if len(self.loss) > 200 and (self.loss[-round(len(self.loss)/100)]-self.loss[-1]) < ((self.loss[0]-self.loss[-1])*1e-4):
                    if not mute:
                        loading_bar._kill()
                        print("Analysis ended due to early stopping.\n")
                    print(self._apply_constraints_AB(A_non_constraint)[0,0])
                    break
        
        
        
        ########## POST ANALYSIS ##########
        Z_f = (X@self._apply_constraints_AB(B_non_constraint).detach().numpy())
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint,p)
        alphas_f = self._calculate_alpha(b_f,p)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (X_tilde_f@self._apply_constraints_AB(B_non_constraint).detach().numpy())
        X_hat_f = self._calculate_X_hat(X_tilde_f,A_f,B_f)
        sigma_f = self._apply_constraints_sigma(sigma_non_constraint).detach().numpy()
        end = timer()
        time = round(end-start,2)
        result = _OAA_result(A_f,B_f,X,n_iter,b_f.detach().numpy(),Z_f,X_tilde_f,Z_tilde_f,X_hat_f,self.loss,K,time,columns,"RBOAA",sigma_f, with_synthetic_data=with_synthetic_data)

        if not mute:
            result._print()
    
        return result