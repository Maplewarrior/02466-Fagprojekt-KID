import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from timeit import default_timer as timer
from scipy.special import softmax
from AA_result_class import _CAA_result

from loading_bar_class import _loading_bar

class _TSAA:
    
    RSS = []    
    
    def _logOdds(self, X):
        Ordinals = range(min(X.flatten()), max(X.flatten()+1))
    
        probs = [(np.count_nonzero(X.flatten() == e))/len(X.flatten()) for e in Ordinals]
        baseline = max(probs)
    
        logvals = [np.log(probs[i]/baseline) for i in range(len(probs))]
        return logvals
    
    def _applySoftmax(self,M):
        return softmax(M)
    
    
    def _projectOrdinals(self, X):
        M, N = X.shape
        
        X_thilde = np.empty((M, N))
        
        theta = self._applySoftmax(self._logOdds(X))
        Ordinals = range(min(X.flatten()), max(X.flatten()+1))
        for i in range(M):
            for j in range(N):
                idx = X[i,j]-1
                X_thilde[i,j] = theta[idx]
                
        return X_thilde
    
    def _error(self, X,B,A):
        return torch.norm(X - X@B@A, p='fro')**2
    
    def _apply_constraints(self, A):
        m = nn.Softmax(dim=0)
        return m(A)
    
    
    ############# Two-step ordinal AA #############
    def _compute_archetypes(self, X, K, n_iter, lr, mute,columns):
        
        ##### Project the data #####
        X = self._projectOrdinals(X)
        
        
        ########## INITIALIZATION ##########
        self.RSS = []
        start = timer()
        if not mute:
            loading_bar = _loading_bar(n_iter, "Conventional Arhcetypal Analysis")
        N, _ = X.T.shape
        Xt = torch.tensor(X,requires_grad=False).float()
        A = torch.autograd.Variable(torch.rand(K, N), requires_grad=True)
        B = torch.autograd.Variable(torch.rand(N, K), requires_grad=True)
        optimizer = optim.Adam([A, B], amsgrad = True, lr = 0.01)


        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(Xt, self._apply_constraints(B), self._apply_constraints(A))
            self.RSS.append(L.detach().numpy())
            L.backward()
            optimizer.step()
            

        ########## POST ANALYSIS ##########
        A_f = self._apply_constraints(A).detach().numpy()
        B_f = self._apply_constraints(B).detach().numpy()
        Z_f = (Xt@self._apply_constraints(B)).detach().numpy()
        X_hat_f = X@B_f@A_f
        end = timer()
        time = round(end-start,2)
        result = _CAA_result(A_f, B_f, X, X_hat_f, n_iter, self.RSS, Z_f, K, time,columns,"TSAA")

        if not mute:
            result._print()

        return result
    