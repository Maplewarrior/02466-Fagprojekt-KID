"""
Experiment to run the model on real data
"""

from OrdinalAA import RB_OAA
from Evaluation_functions import archetype_correlation,NMI,ResponsBiasCompereson
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
times = 10

save=True

#import data
data_all=pd.read_csv(r"C:\Users\Andre\OneDrive - Danmarks Tekniske Universitet\Bachelor project\Schmidt_et_al_2021_Latent_profile_analysis_of_human_values_SUPPL\VB_LPA\Data\ESS8_data.csv")
keys=["SD1","PO1","UN1","AC1","SC1","ST1","CO1","UN2","TR1","HD1","SD2","BE1","AC2","SC2","ST2","CO2","PO2","BE2","UN3","TR2","HD2"]
data=data_all[keys]

savedir = r"C:\Users\Andre\OneDrive - Danmarks Tekniske Universitet\Bachelor project\18Types"

Astart=18
epokes=1250


loss_log_OAA=pd.DataFrame(columns=range(times), index=range(Astart, len(keys)))
loss_log_RB_OAA=pd.DataFrame(columns=range(times), index=range(Astart, len(keys)))

intermodel_ACC=pd.DataFrame(columns=["Mean archetype correlation OAA","Mean archetype correlation RB OAA"],index=range(Astart,len(keys)))
intermodel_NMI=pd.DataFrame(columns=["Mean S NMI OAA","Mean S NMI RB OAA"],index=range(Astart,len(keys)))
intermodel_RBC=pd.DataFrame(columns=["Mean beta difference OAA","Mean beta difference RB OAA"],index=range(Astart,len(keys)))
permutatuions=itertools.permutations(range(times))

for k in range(Astart,len(keys)):
    Slist1=[]
    Alist1=[]
    Blist1=[]
    Slist2=[]
    Alist2=[]
    Blist2=[]

    run_loss_OAA = pd.DataFrame(columns=range(times), index=range(epokes))
    run_loss_RB_OAA = pd.DataFrame(columns=range(times), index=range(epokes))

    for i in range(times):
        summery_RB_OAA, result_RB_OAA, summery_OAA, result_OAA = RB_OAA(data, k, epokes=epokes, learning_rata=0.01, verbose=False, save=save,
                                                                        savedir=savedir, fileName=f"K{k}_sample{i}")
        print(f"K{k} iteration {i}")
        loss_log_OAA.loc[k, i]=summery_OAA["loss"]
        loss_log_RB_OAA.loc[k, i]=summery_RB_OAA["loss"]

        Slist1.append(result_OAA["S"])
        Alist1.append(result_OAA["A"])
        Slist2.append(result_RB_OAA["S"])
        Alist2.append(result_RB_OAA["A"])
        Blist1.append(result_OAA["beta"])
        Blist2.append(result_RB_OAA["beta"])

        run_loss_OAA.iloc[:, i] = result_OAA["loss_log"]
        run_loss_RB_OAA.iloc[:, i] = result_RB_OAA["loss_log"]


    intermodel_ACC.loc[k,"Mean archetype correlation OAA"]=np.mean([archetype_correlation(Alist1[permutation[0]],Alist1[permutation[1]])[2] for permutation in itertools.permutations(range(times),2)])
    intermodel_ACC.loc[k,"Mean archetype correlation RB OAA"]=np.mean([archetype_correlation(Alist2[permutation[0]],Alist2[permutation[1]])[2] for permutation in itertools.permutations(range(times),2)])

    intermodel_NMI.loc[k,"Mean S NMI OAA"]=np.mean([NMI(Slist1[permutation[0]],Slist1[permutation[1]]) for permutation in itertools.permutations(range(times),2)])
    intermodel_NMI.loc[k,"Mean S NMI RB OAA"]=np.mean([NMI(Slist2[permutation[0]],Slist2[permutation[1]]) for permutation in itertools.permutations(range(times),2)])


    intermodel_RBC.loc[k,"Mean beta difference OAA"]=np.mean([ResponsBiasCompereson(Blist1[permutation[0]],Blist1[permutation[1]])[1] for permutation in itertools.permutations(range(times),2)])
    intermodel_RBC.loc[k,"Mean beta difference RB OAA"]=np.mean([ResponsBiasCompereson(Blist2[permutation[0]],Blist2[permutation[1]])[1] for permutation in itertools.permutations(range(times),2)])


    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

    loss_log_OAA.mean(1).plot(c="blue", ax=ax1, legend=True,ylabel="loss",xlabel="Number of archetypes")
    loss_log_RB_OAA.mean(1).plot(c="red", ax=ax1, legend=True)
    ax1.legend(["OAA loss","RB OAA loss"])

    loss_log_OAA.plot(style="o", c="blue", ax=ax1, legend=False, title="Loss")
    loss_log_RB_OAA.plot(style="o", c="red", ax=ax1, legend=False)

    intermodel_ACC.plot(ax=ax2,ylim=(0,1),color=["blue","red"],title="intermodel agrement", xlabel="Number of archetypes")
    intermodel_NMI.plot(ax=ax2,ylim=(0,1),color=["blue","red"],style=":")
    intermodel_RBC.plot(ax=ax2, ylim=(0, 1), color=["blue", "red"], style="--")


    plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    run_loss_OAA.plot(ax=ax1, ylabel="loss", xlabel="Epokes", title="Ordinal Archetype analysis (Initialising model)")
    run_loss_RB_OAA.plot(ax=ax2, ylabel="loss", xlabel="Epokes", title="Respons bias OAA")
    plt.show()
    if save:
        intermodel_ACC.to_csv(os.path.join(savedir,f"K{k}intermodel_ACC"))
        intermodel_NMI.to_csv(os.path.join(savedir, f"K{k}intermodel_NMI"))
        intermodel_RBC.to_csv(os.path.join(savedir, f"K{k}intermodel_RBC"))
        run_loss_OAA.to_csv(os.path.join(savedir,f"K{k}OAA_loss"))
        run_loss_RB_OAA.to_csv(os.path.join(savedir, f"K{k}_OAA_loss"))