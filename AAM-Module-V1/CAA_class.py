########## IMPORTS ##########
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer

from loading_bar_class import _loading_bar
from AA_result_class import _CAA_result


########## CONVENTIONAL ARCHETYPAL ANALYSIS CLASS ##########
class _CAA:
    
    RSS = []

    def _error(self, X,B,A):
        return torch.norm(X - X@B@A, p='fro')**2
    
    def _apply_constraints(self, A):
        m = nn.Softmax(dim=0)
        return m(A)
    
    def _compute_archetypes(self, X, K, n_iter, lr, mute,columns):

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
        result = _CAA_result(A_f, B_f, X, X_hat_f, n_iter, self.RSS, Z_f, K, time,columns,"CAA")

        if not mute:
            result._print()

        return result