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

        Ordinals = range(int(min(X.flatten())), int(max(X.flatten()+1)))
        probs = [(np.count_nonzero(X.flatten() == e))/len(X.flatten()) for e in Ordinals]
        baseline = max(probs)
        logvals = [np.log(probs[i]/baseline) for i in range(len(probs))]

        return logvals
    
    def _applySoftmax(self,M):
        return softmax(M)
    
    def _convertScores(self, X):
        
        Ordinals = range(int(min(X.flatten())), int(max(X.flatten()+1)))
        thetas = self._applySoftmax(self._logOdds(X))
        scores = [1+((k+1)-1)*thetas[k] for k in range(len(Ordinals))]
        
        return scores
        
    def _projectOrdinals(self, X):
        
        M, N = X.shape
        X_hat = np.empty((M, N))
        
        scores = self._convertScores(X)
        
        for i in range(M):
            for j in range(N):
                idx = X[i,j]-1
                X_hat[i,j] = scores[idx]
        return X_hat
    
    def _error(self, X,B,A):
        return torch.norm(X - X@B@A, p='fro')**2
    
    def _apply_constraints(self, A):
        m = nn.Softmax(dim=0)
        return m(A)
    
    ############# Two-step ordinal AA #############
    def _compute_archetypes(self, X, K, p, n_iter, lr, mute,columns, with_synthetic_data = False, early_stopping = False):
        
        ##### Project the data #####
        # Xt = torch.tensor(X, dtype = torch.long)
        
        X_hat = self._projectOrdinals(X)
        X_hat = torch.tensor(X_hat)
        
        ########## INITIALIZATION ##########
        self.RSS = []
        start = timer()
        if not mute:
            loading_bar = _loading_bar(n_iter, "Conventional Arhcetypal Analysis")
        N, _ = X.T.shape
        A = torch.autograd.Variable(torch.rand(K, N), requires_grad=True)
        B = torch.autograd.Variable(torch.rand(N, K), requires_grad=True)
        optimizer = optim.Adam([A, B], amsgrad = False, lr = lr)
        

        ########## ANALYSIS ##########
        for i in range(n_iter):
            if not mute:
                loading_bar._update()
            optimizer.zero_grad()
            L = self._error(X_hat, self._apply_constraints(B).double(), self._apply_constraints(A).double())
            self.RSS.append(L.detach().numpy())
            L.backward()
            optimizer.step()
            
            ########## EARLY STOPPING ##########
            if i % 25 == 0 and early_stopping:
                if len(self.RSS) > 200 and (self.RSS[-round(len(self.RSS)/100)]-self.RSS[-1]) < ((self.RSS[0]-self.RSS[-1])*1e-4):
                    if not mute:
                        loading_bar._kill()
                        print("Analysis ended due to early stopping.\n")
                    break
            

        ########## POST ANALYSIS ##########
        A_f = self._apply_constraints(A).detach().numpy()
        B_f = self._apply_constraints(B).detach().numpy()
        Z_f = X @ self._apply_constraints(B).detach().numpy()
        
        X_hat_f = X_hat.detach().numpy()
        end = timer()
        time = round(end-start,2)
        result = _CAA_result(A_f, B_f, X, X_hat_f, n_iter, self.RSS, Z_f, K, p, time,columns,"TSAA", with_synthetic_data = with_synthetic_data)
        if not mute:
            result._print()

        return result
    