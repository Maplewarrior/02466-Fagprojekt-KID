"""
Experiment to test model the Ordinal Archetype models ability to learn a new scale without anotaion bias
"""

from OrdinalAA import Ordinal_AA, RB_OAA
from AAordinalSampler import OrdinalSampler
from Evaluation_functions import archetype_correlation,NMI,ResponsBiasCompereson
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

times=2



bias=False
betaParm =1000
alphaFalse=[0,1,2,3,4]
K=4
sigma_list = [0.01] # sigma = -4.6
N=10000
M=21
epokes=1000

save=False

savedir=r"C:\Users\micha\OneDrive\Skrivebord\02466-Fagprojekt-KID\Code\NewScaleExperiment"


loss_log_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
NMI_log_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
RBC_log_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
RBC_log_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
A_correlation_OAA=pd.DataFrame(columns=range(times), index=sigma_list)


run_loss_OAA = pd.DataFrame(columns=range(times), index=range(epokes))


for sigma in sigma_list:
    print("inside")

    dataSample,dataTrue,ATrue,STrue,beta=OrdinalSampler(N=N,M=M,sigma=sigma,K=K,alphaFalse=alphaFalse,betaParm=betaParm,bias=bias)
    print("Created synthetic data")
    label,count=np.unique(dataSample, return_counts=True)
    plt.bar(label,count)
    plt.title(f"Simulated data densety sigma={sigma}")
    plt.show()

    saveDict = {"beta": beta.tolist(),
                "sigma": sigma,
                "A":ATrue.tolist(),
                "X_observed": dataSample.tolist(),
                "X_true": dataTrue.tolist(),
                "S_True": STrue.tolist()}

    if save:
        with open(os.path.join(savedir, f"Sigma{sigma}true"), "w") as file:
            json.dump(saveDict, file)
    print("SHAPE OF DATA:", dataSample.shape)
    print("Further")
    for i in range(times):
        summery_OAA, result_OAA=Ordinal_AA(dataSample, K, learning_rata=0.01, epokes=epokes, verbose=False, save=save, savedir=savedir, fileName=f"sigma{sigma}sample{i}")
        print("iter number: ",i)
        RBC_log_OAA.loc[sigma, i] = ResponsBiasCompereson(beta, result_OAA["beta"])[1]

        loss_log_OAA.loc[sigma, i]=summery_OAA["loss"]

        NMI_log_OAA.loc[sigma, i]=NMI(STrue, result_OAA["S"])

        A_correlation_OAA.loc[sigma, i]=archetype_correlation(ATrue, result_OAA["A"])[2]

        run_loss_OAA.iloc[:, i] = result_OAA["loss_log"]

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))

    loss_log_OAA.mean(1).plot(c="blue", ax=ax1, legend=True, ylabel="loss", xlabel="noise")
    ax1.legend(["loss"])

    loss_log_OAA.plot(style="o", c="blue", ax=ax1, legend=False, title="Loss", label="Model 1 loss")

    NMI_log_OAA.mean(1).plot(c="blue", ax=ax2, xlabel="noise")
    RBC_log_OAA.mean(1).plot(style="--", c="blue", ax=ax2)
    A_correlation_OAA.mean(1).plot(style=":", c="blue", ax=ax2)
    ax2.legend(["NMI","Archetype correlation","scale error"])

    RBC_log_OAA.plot(style="*", c="blue", ax=ax2, legend=False)
    NMI_log_OAA.plot(style="o", c="blue", ax=ax2, ylim=(0, 1), title="Reconstruction", legend=False)
    A_correlation_OAA.plot(style="+", c="blue", ax=ax2, legend=False)
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    run_loss_OAA.plot(ax=ax1, ylabel="loss", xlabel="Epokes", title="Ordinal Archetype analysis Training loss")
    plt.show()

    if save:
        loss_log_OAA.to_csv(os.path.join(savedir, "OAAloss_log"))
        NMI_log_OAA.to_csv(os.path.join(savedir, "OAA_NMI_log"))
        A_correlation_OAA.to_csv(os.path.join(savedir, "OAA_Acor_log"))


