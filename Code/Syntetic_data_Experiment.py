"""
Experiment to test model performance on syntetic data.
"""

from OrdinalAA import Ordinal_AA, RB_OAA
#from OrdinalAAbias1 import Ordinal_AA
from AAordinalSampler import OrdinalSampler
from Evaluation_functions import archetype_correlation,NMI,ResponsBiasCompereson
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

times=3

betaParm = 5

bias=True

alphaFalse=[0,1,2,3,4]
K=4
sigma_list=np.arange(0,0.5,0.05).round(2)
N=10000
M=21
epokes=1000

save=True

savedir=r"C:\Users\Andre\OneDrive - Danmarks Tekniske Universitet\Bachelor project\Simulated5"


loss_log_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
loss_log_RB_OAA=pd.DataFrame(columns=range(times), index=sigma_list)

NMI_log_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
NMI_log_RB_OAA=pd.DataFrame(columns=range(times), index=sigma_list)

RBC_log_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
RBC_log_RB_OAA=pd.DataFrame(columns=range(times), index=sigma_list)

A_correlation_OAA=pd.DataFrame(columns=range(times), index=sigma_list)
A_correlation_RB_OAA=pd.DataFrame(columns=range(times), index=sigma_list)

run_loss_OAA = pd.DataFrame(columns=range(times), index=range(epokes))
run_loss_RB_OAA = pd.DataFrame(columns=range(times), index=range(epokes))

RBC_baceline=pd.Series(index=sigma_list,dtype="float64")

for sigma in sigma_list:


    dataSample,dataTrue,ATrue,STrue,beta=OrdinalSampler(N=N,M=M,sigma=sigma,K=K,alphaFalse=alphaFalse,betaParm=betaParm,bias=bias)
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

    RBC_baceline.loc[sigma]=ResponsBiasCompereson(beta, [0,0.2,0.4,0.6,0.8,1])[1]

    for i in range(times):
        summery_RB_OAA, result_RB_OAA, summery_OAA, result_OAA=RB_OAA(dataSample, K, learning_rata=0.01, epokes=epokes, verbose=False, save=save, savedir=savedir, fileName=f"sigma{sigma}sample{i}")

        loss_log_OAA.loc[sigma, i]=summery_OAA["loss"]
        loss_log_RB_OAA.loc[sigma, i]=summery_RB_OAA["loss"]

        NMI_log_OAA.loc[sigma, i]=NMI(STrue, result_OAA["S"])
        NMI_log_RB_OAA.loc[sigma, i]=NMI(STrue, result_RB_OAA["S"])

        RBC_log_OAA.loc[sigma, i]=ResponsBiasCompereson(beta, result_OAA["beta"])[1]
        RBC_log_RB_OAA.loc[sigma, i]=ResponsBiasCompereson(beta, result_RB_OAA["beta"])[1]

        A_correlation_OAA.loc[sigma, i]=archetype_correlation(ATrue, result_OAA["A"])[2]
        A_correlation_RB_OAA.loc[sigma, i]=archetype_correlation(ATrue, result_RB_OAA["A"])[2]

        run_loss_OAA.iloc[:, i] = result_OAA["loss_log"]
        run_loss_RB_OAA.iloc[:, i] = result_RB_OAA["loss_log"]

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,10))

    loss_log_OAA.mean(1).plot(c="blue", ax=ax1, legend=True, ylabel="loss", xlabel="noise")
    loss_log_RB_OAA.mean(1).plot(c="red", ax=ax1, legend=True)
    ax1.legend(["OAA loss","RB OAA loss"])

    loss_log_OAA.plot(style="o", c="blue", ax=ax1, legend=False, title="Loss")
    loss_log_RB_OAA.plot(style="o", c="red", ax=ax1, legend=False)

    NMI_log_OAA.mean(1).plot(c="blue", ax=ax2, ylabel="%", xlabel="noise")
    NMI_log_RB_OAA.mean(1).plot(c="red", ax=ax2)
    A_correlation_OAA.mean(1).plot(style=":", c="blue", ax=ax2)
    A_correlation_RB_OAA.mean(1).plot(style=":", c="red", ax=ax2)
    RBC_log_OAA.mean(1).plot(style="--", c="blue", ax=ax2)
    RBC_log_RB_OAA.mean(1).plot(style="--", c="red", ax=ax2)
    RBC_baceline.plot(style="-",c="black",ax=ax2)

    ax2.legend(["OAA NMI","RB_OAA NMI","OAA Archetype correlation","RB OAA Archetype correlation","Mean beta difference OAA","Mean beta difference RB OAA","baseline beta difference"])

    NMI_log_OAA.plot(style="o", c="blue", ax=ax2, ylim=(0, 1), title="Reconstruction", legend=False)
    NMI_log_RB_OAA.plot(style="o", c="red", ax=ax2, legend=False)

    RBC_log_OAA.plot(style="*", c="blue", ax=ax2, legend=False)
    RBC_log_RB_OAA.plot(style="*", c="red", ax=ax2, legend=False)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    run_loss_OAA.plot(ax=ax1, ylabel="loss", xlabel="Epokes", title="Ordinal Archetype analysis (Initialising model)")
    run_loss_RB_OAA.plot(ax=ax2, ylabel="loss", xlabel="Epokes", title="Respons bias OAA")
    plt.show()

    if save:
        loss_log_OAA.to_csv(os.path.join(savedir, "OAAloss_log"))
        loss_log_RB_OAA.to_csv(os.path.join(savedir, "RB_OAAloss_log"))
        NMI_log_OAA.to_csv(os.path.join(savedir, "OAA_NMI_log"))
        NMI_log_RB_OAA.to_csv(os.path.join(savedir, "RB_OAA_NMI_log"))
        A_correlation_OAA.to_csv(os.path.join(savedir, "OAA_Acor_log"))
        A_correlation_RB_OAA.to_csv(os.path.join(savedir, "RB_OAAAcor_log"))

