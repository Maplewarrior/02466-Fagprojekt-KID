


import torch
import numpy as np
from synthetic_data_class import _synthetic_data



def OrdinalSampler(N,M,alphaFalse,sigma,K,achetypeParm=1,betaParm=1,bias=True):
    # torch.manual_seed(0)
    J=len(alphaFalse)

    if isinstance(betaParm,(int,float)):
        #Make betaParm a list
        betaParm=torch.empty((J)).fill_(betaParm)

    elif isinstance(betaParm,(list)):
        betaParm= torch.tensor(betaParm,dtype=torch.float)

    #Find alpha True
    beta=torch.cat((torch.tensor([0]),torch.cumsum(betaParm/(torch.sum(betaParm)),dim=0)),dim=0)
    alphaTrue = (beta[:J] + beta[1:J + 1]) / 2




    if bias:
        #Sampel beta values from Dirichlet distibution
        betaSampler=torch.distributions.dirichlet.Dirichlet(betaParm)
        beta = torch.empty((N,J+1))
        beta[:,1:J]=torch.stack([betaSampler.sample() for n in range(N)]).cumsum(dim=1)[:,:-1]

    else:
        beta=beta.repeat(N,1)

    #Extend edge catagory to infint
    beta[:,0] = -float("Inf")
    beta[:,J] = float("Inf")

    #Make an achtype sampler
    archetypyeSample=torch.distributions.dirichlet.Dirichlet(torch.empty((K)).fill_(achetypeParm))

    #Reconstuction matric is which compination of types each person is
    S=torch.stack([archetypyeSample.sample() for n in range(N)])


    #Ordinal archtypes
    archetypes = torch.floor(torch.rand((K, M)) * J).type(dtype=torch.long)
    achtypesOrdinal = alphaTrue[archetypes]
    Xhat = torch.matmul(S, achtypesOrdinal)

    P = torch.empty((N, M, J))

    if sigma==0:
        index=torch.stack([torch.stack([beta[n,j]<Xhat[n,:] for n in range(N)]) for j in range(1,J)]).sum(0)
    else:

        Zeta = torch.stack([torch.stack([(beta[n,j]-Xhat[n,:])/sigma for n in range(N)]) for j in range(J+1)])

        for j in range(J):
            P[:,:,j]=torch.distributions.normal.Normal(0, 1).cdf(Zeta[j+1])- torch.distributions.normal.Normal(0, 1).cdf(Zeta[j])

        index=torch.distributions.Categorical(P).sample()

    alphaFalse = np.array(alphaFalse)
    Xnew=alphaFalse[index]

    return Xnew,Xhat.numpy(),archetypes.numpy(),S.numpy(),beta.numpy()
