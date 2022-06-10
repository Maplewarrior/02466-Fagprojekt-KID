"""
Implementation of OAA and RB-OAA
"""

import torch
import numpy as np
torch.manual_seed(0)
import time
import json
import os
import matplotlib.pyplot as plt

def Ordinal_AA(data, K, epokes, learning_rata, device="auto", TildeOutput=False, verbose=False, save=False, savedir="", fileName=""):
    start_time = time.time()
    if device=="auto":
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"
    if verbose: print('Using device:', device)

    #Preprosses X into new ordinals (Pre prosing)
    unique=np.unique(data)
    new_arrangement=np.arange(len(unique))
    map_func = np.vectorize(lambda x: new_arrangement[unique==x][0])
    X=torch.tensor(map_func(data),dtype=torch.long,device=device)

    N, M = X.size()

    #Find the number of ordinals
    J=len(unique)
    gamma = torch.randn(J,device=device)
    gamma.requires_grad = True

    sigmaTilde = torch.rand(1,dtype=torch.float64,device=device)
    sigmaTilde.requires_grad = True


    beta = torch.empty(J+1,device=device)
    beta[0] = 0

    #Random initialization
    i=np.random.randint(0,N,K,dtype=np.int64)

    #Initialize C~
    Ctilde=torch._sparse_csr_tensor(torch.tensor(range(K+1)),torch.tensor(i),torch.ones(K),(K,N)).to_dense().to(device=device)
    Ctilde = Ctilde*np.log(N*2)
    Ctilde.requires_grad=True

    #Initilize s~
    Stilde=torch.rand(N,K).to(device=device)
    Stilde.requires_grad=True

    optimiser = torch.optim.Adam([Ctilde, Stilde,gamma,sigmaTilde], lr=learning_rata)

    #Set up training loop
    loss_log=[]
    loss_best=float("Inf")


    for epoke in range(epokes):


        sigma = torch.log(1 + torch.exp(sigmaTilde))

        beta[1:J+1] = torch.cumsum(torch.nn.functional.softmax(gamma.clone(),dim=0), dim=0)
        alpha = (beta[:J] + beta[1:J+1]) / 2

        #Map X to Xtilde
        Xtilde=alpha[X]

        #Apply softmax
        C = torch.nn.functional.softmax(Ctilde,dim=1)
        S = torch.nn.functional.softmax(Stilde, dim=1)

        #Perform AA
        A = torch.matmul(C,Xtilde)
        Xhat=torch.matmul(S,A)

        #Find zeta values
        zetaNext = (beta[X+1] - Xhat)/sigma
        zetaPrev = (beta[X] - Xhat)/sigma


        zetaNext[X==J-1] = float("Inf")
        zetaPrev[X == 0] = -float("Inf")


        #Do phi(zeta1)-phi(zeta2)
        logP= -torch.log((torch.distributions.normal.Normal(0, 1).cdf(zetaNext)- torch.distributions.normal.Normal(0, 1).cdf(zetaPrev))+10E-10) #add small number to avoid underflow.

        loss = torch.sum(logP)

        loss_log.append(loss.detach().cpu().item())

        #Perfome gradient step
        optimiser.zero_grad()
        loss.backward(retain_graph=True)
        optimiser.step()

        if loss<loss_best:
            loss_best = loss
            best_epoke = epoke
            result = {"loss_log": loss_log,
                      "A": A.detach().cpu().tolist(),
                      "alpha": alpha.detach().cpu().tolist(),
                      "beta": beta.detach().cpu().tolist(),
                      "Xhat": Xhat.detach().cpu().tolist(),
                      "Xtilde": Xtilde.detach().cpu().tolist(),
                      "C": C.detach().cpu().tolist(),
                      "S": S.detach().cpu().tolist()}


        if verbose:
            print(f"itereration {best_epoke} time {time.time()-start_time} Loss {loss_best} sigma={sigma.item()} alpha={alpha.tolist()} cuts={(beta).tolist()}")

    print(f"itereration {best_epoke} time {time.time() - start_time} Loss {loss_best}")

    summery = {"loss": loss_best.detach().cpu().item(), "Sigma": sigma.detach().cpu().item(), "best_epokes": best_epoke, "RunTime": time.time() - start_time}

    if save:
        with open(os.path.join(savedir,fileName+"OAA_Summery"),"w") as file:
            json.dump(summery,file)

        with open(os.path.join(savedir,fileName+"OAA_Result"),"w") as file:
            json.dump(result,file)



    if TildeOutput:
        return gamma.detach(),sigmaTilde.detach(),Ctilde.detach(),Stilde.detach(),X, summery,result
    else:
        return summery,result

def RB_OAA(data, K, epokes, learning_rata, device="auto", verbose=False, save=False, savedir="", fileName=""):
    start_time = time.time()
    if device=="auto":
        if torch.cuda.is_available():
            device="cuda"
        else:
            device="cpu"


    gamma,sigmaTilde,Ctilde,Stilde,X,summery_StandardModel,result_StandardModel=Ordinal_AA(
        data, K, epokes=epokes,learning_rata=learning_rata, device=device, TildeOutput=True,
        verbose=verbose, save=save,savedir=savedir, fileName=fileName)

    J=len(gamma)
    N, M = X.size()

    #Duplicate paremeter
    gamma=gamma.repeat(N,1)
    gamma.requires_grad=True

    sigmaTilde=sigmaTilde.repeat(N,1)
    sigmaTilde.requires_grad=True

    Ctilde.requires_grad=True
    Stilde.requires_grad=True

    optimiser = torch.optim.Adam([Ctilde, Stilde,gamma,sigmaTilde], lr=learning_rata)

    beta = torch.empty((N,J+1),device=device)
    beta[:,0] = 0


    #Set up training loop
    loss_log=[]
    loss_best=float("Inf")

    for epoke in range(epokes):

        sigma = torch.log(1 + torch.exp(sigmaTilde))

        #Find alpha and cuts
        beta[:, 1:J+1] = torch.cumsum(torch.nn.functional.softmax(gamma.clone(),dim=1),dim=1)
        alpha = (beta[:,0:J] + beta[:,1:J+1]) / 2

        #Map X to Xtilde
        XTilde=torch.gather(alpha,1,X) #unike værdier af X

        #Aply softmax
        C = torch.nn.functional.softmax(Ctilde,dim=1)
        S = torch.nn.functional.softmax(Stilde, dim=1)

        #Perform AA
        A = torch.matmul(C,XTilde)
        Xhat=torch.matmul(S,A)

        #Find zeta values
        zetaNext = (torch.gather(beta,1,X+1)-Xhat)/sigma
        zetaPrev = (torch.gather(beta,1,X)-Xhat)/sigma


        #Set all Zeta values of first ordinal to -infinet and all values of last ordinal to infinet to make sure everyone fall intor the categories.
        zetaNext[X==J-1] = float("Inf")
        zetaPrev[X == 0] = -float("Inf")

        #Do phi(zeta1)-phi(zeta2)
        logP= -torch.log((torch.distributions.normal.Normal(0, 1).cdf(zetaNext)-torch.distributions.normal.Normal(0, 1).cdf(zetaPrev))+10E-10) #add small number to avoid underflow.

        loss = torch.sum(logP)


        loss_log.append(loss.detach().cpu().item())

        #Perfome gradient step
        optimiser.zero_grad()
        loss.backward(retain_graph=True)
        optimiser.step()

        if loss<loss_best:
            loss_best=loss
            best_epoke=epoke
            result = {"loss_log": loss_log,
                      "A": A.detach().cpu().tolist(),
                      "alpha": alpha.detach().cpu().tolist(),
                      "beta": beta.detach().cpu().tolist(),
                      "Xhat": Xhat.detach().cpu().tolist(),
                      "Xtilde": XTilde.detach().cpu().tolist(),
                      "C": C.detach().cpu().tolist(),
                      "S": S.detach().cpu().tolist()}

        if verbose:
            print(f"itereration {epoke} time {time.time()-start_time} Loss {loss}")

    print(f"itereration {best_epoke} time {time.time() - start_time} Loss {loss_best}")

    summery = {"loss": loss_best.detach().cpu().item(), "Sigma": torch.mean(sigma).detach().cpu().item(),"best_epokes": best_epoke, "RunTime": time.time() - start_time}




    if save:
        with open(os.path.join(savedir,fileName+"RB_Summery"),"w") as file:
            json.dump(summery,file)

        with open(os.path.join(savedir,fileName+"RB_Result"),"w") as file:
            json.dump(result,file)

    return summery,result,summery_StandardModel,result_StandardModel

def plot_loss(loss):
    plt.plot(loss)
    plt.ylabel("loss")
    plt.xlabel("Epokes")
    plt.show()