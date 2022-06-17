########## IMPORTS ##########
from re import T
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from AA_result_class import _OAA_result
from loading_bar_class import _loading_bar
from CAA_class import _CAA



########## ORDINAL ARCHETYPAL ANALYSIS CLASS ##########
class _OAA:

    ########## HELPER FUNCTION // EARLY STOPPING ##########
    def _early_stopping(self):
        next_imp = self.loss[-round(len(self.loss)/100)]-self.loss[-1]
        prev_imp = (self.loss[0]-self.loss[-1])*1e-4
        return next_imp < prev_imp

    ########## HELPER FUNCTION // A AND B ##########
    def _apply_constraints_AB(self,A):
        m = nn.Softmax(dim=1)
        return m(A)

    ########## HELPER FUNCTION // BETAS ##########
    def _apply_constraints_beta(self,b): 
        m = nn.Softmax(dim=0)
        return torch.cumsum(m(b), dim=0)[:len(b)-1]

    ########## HELPER FUNCTION // SIGMA ##########
    def _apply_constraints_sigma(self,sigma):
        m = nn.Softplus()
        return m(sigma)

    ########## HELPER FUNCTION // ALPHA ##########
    def _calculate_alpha(self,b):
        b_j = torch.cat((torch.tensor([0.0]),b),0)
        b_j_plus1 = torch.cat((b,torch.tensor([1.0])),0)
        alphas = (b_j_plus1+b_j)/2
        return alphas

    ########## HELPER FUNCTION // X_tilde ##########
    def _calculate_X_tilde(self,X,alphas):
        X_tilde = alphas[X.long()-1]
        return X_tilde

    ########## HELPER FUNCTION // X_hat ##########
    def _calculate_X_hat(self,X_tilde,A,B):
        Z = B @ X_tilde
        X_hat = A @ Z
        return X_hat
    
    ########## HELPER FUNCTION // LOSS ##########
    def _calculate_loss(self,Xt, X_hat, b, sigma):
        
        pad = nn.ConstantPad1d(1, 0)
        b = pad(b)
        b[-1] = 1.0

        z_next = (b[Xt] - X_hat)/sigma
        z_prev = (b[Xt-1] - X_hat)/sigma
        z_next[Xt == len(b)+1] = np.inf
        z_prev[Xt == 1] = -np.inf

        P_next = torch.distributions.normal.Normal(0, 1).cdf(z_next)
        P_prev = torch.distributions.normal.Normal(0, 1).cdf(z_prev)
        neg_logP = -torch.log(( P_next - P_prev ) +1e-10)
        loss = torch.sum(neg_logP)

        return loss

    ########## HELPER FUNCTION // ERROR ##########
    def _error(self,Xt,A_non_constraint,B_non_constraint,b_non_constraint,sigma_non_constraint):

        A = self._apply_constraints_AB(A_non_constraint)
        B = self._apply_constraints_AB(B_non_constraint)
        b = self._apply_constraints_beta(b_non_constraint)
        sigma = self._apply_constraints_sigma(sigma_non_constraint)
        alphas = self._calculate_alpha(b)
        
        X_tilde = self._calculate_X_tilde(Xt,alphas)
        X_hat = self._calculate_X_hat(X_tilde,A,B)

        loss = self._calculate_loss(Xt, X_hat, b, sigma)
        
        return loss
        

    ########## COMPUTE ARCHETYPES FUNCTION OF OAA ##########
    def _compute_archetypes(
        self, 
        X, 
        K, 
        p, 
        n_iter, 
        lr, 
        mute, 
        columns, 
        with_synthetic_data = False, 
        early_stopping = False, 
        for_hotstart_usage = False):


        ########## INITIALIZATION ##########
        self.N, self.M = len(X.T), len(X.T[0,:])
        Xt = torch.tensor(X.T, dtype = torch.long)
        self.loss = []
        start = timer()

        A_non_constraint = torch.autograd.Variable(torch.randn(self.N, K), requires_grad=True)
        B_non_constraint = torch.autograd.Variable(torch.randn(K, self.N), requires_grad=True)
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
                if len(self.loss) > 200 and self._early_stopping():
                    if not mute:
                        loading_bar._kill()
                        print("Analysis ended due to early stopping.\n")
                    break
            
        
        ########## POST ANALYSIS ##########
        A_f = self._apply_constraints_AB(A_non_constraint).detach().numpy()
        B_f = self._apply_constraints_AB(B_non_constraint).detach().numpy()
        b_f = self._apply_constraints_beta(b_non_constraint)
        alphas_f = self._calculate_alpha(b_f)
        X_tilde_f = self._calculate_X_tilde(Xt,alphas_f).detach().numpy()
        Z_tilde_f = (self._apply_constraints_AB(B_non_constraint).detach().numpy() @ X_tilde_f)
        sigma_f = self._apply_constraints_sigma(sigma_non_constraint).detach().numpy()
        X_hat_f = self._calculate_X_hat(X_tilde_f,A_f,B_f)
        end = timer()
        time = round(end-start,2)
        Z_f = B_f @ X_tilde_f


        ########## CREATE RESULT INSTANCE ##########
        result = _OAA_result(
            A_f.T,
            B_f.T,
            X,
            n_iter,
            b_f.detach().numpy(),
            Z_f.T,
            X_tilde_f.T,
            Z_tilde_f.T,
            X_hat_f.T,
            self.loss,
            K,
            p,
            time,
            columns,
            "OAA",
            sigma_f,
            with_synthetic_data=with_synthetic_data)

        if not mute:
            result._print()
        

        ########## RETURN RESULT IF REGULAR, RETURN MATRICIES IF HOTSTART USAGE ##########
        if not for_hotstart_usage:
            return result
        else:
            A_non_constraint_np = A_non_constraint.detach().numpy()
            B_non_constraint_np = B_non_constraint.detach().numpy()
            sigma_non_constraint_np = sigma_non_constraint.detach().numpy()
            b_non_constraint_np = b_non_constraint.detach().numpy()
            return A_non_constraint_np, B_non_constraint_np, sigma_non_constraint_np, b_non_constraint_np
