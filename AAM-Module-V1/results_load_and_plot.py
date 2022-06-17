### Script for loading and plotting results ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import flatten
import os

df = pd.read_csv('full_result_dataset.csv')

attributeNames = list(df.columns)
# print(attributeNames)

# Sorting the values in the df
df = df.sort_values(["sigma", "a_param", "b_param", "synthetic_k", "sigma_std", "rep"], ascending = [True, True, True, True, True, True])


# Find the number of unique parameter values
sigmas = list(df["sigma"].unique())
a_params = list(df["a_param"].unique())
b_params = list(df["b_param"].unique())
synthetic_ks = list(df["synthetic_k"].unique())
analysis_ks = list(df["analysis_k"].unique())
n_reps = int(max(list(df["rep"].unique())))
sigma_stds = list(df["sigma_std"].unique())
AA_types = list(df["AA_type"].unique())
metrics = ['loss', 'NMI', 'MCC', 'BDM', 'Est. sigma']

#%%
data_dict = {}

def isNaN(num):
    return num!= num



for a_param in a_params:
    if not a_param in data_dict:
        data_dict[a_param] = {}
        
    for b_param in b_params:
        if not b_param in data_dict[a_param]:
            data_dict[a_param][b_param] = {}
    
        for k in synthetic_ks:
            if not k in data_dict[a_param][b_param]:
                data_dict[a_param][b_param][k] = {}
                
                for s_std in sigma_stds:
                    if not s_std in data_dict[a_param][b_param][k]:
                        data_dict[a_param][b_param][k][s_std] = {}
                        
                    
                    for type in AA_types:
                        if not type in data_dict[a_param][b_param][k][s_std]:
                            data_dict[a_param][b_param][k][s_std][type] = {}
                            
                            for k_anal in analysis_ks:
                                if not k_anal in data_dict[a_param][b_param][k][s_std][type]:
                                    data_dict[a_param][b_param][k][s_std][type][k_anal] = {}
                        
                        
                                metric_dict = {"loss":[], "NMI": [], "MCC": [], "BDM": [], "Est. sigma": []}
                                for metric in metrics:
                                    if not metric in data_dict[a_param][b_param][k][s_std][type][k_anal]:
                                        data_dict[a_param][b_param][k][s_std][type][k_anal][metric] = {}
                                        
                                    for sigma in sigmas:
                                        if not sigma in data_dict[a_param][b_param][k][s_std][type][k_anal][metric]:
                                            data_dict[a_param][b_param][k][s_std][type][k_anal][metric][sigma] = []
                                        
                                        
                                            
                                        mask = df.loc[(df['a_param'] == a_param) & (df['b_param'] == b_param) 
                                               & (df['synthetic_k'] == k) & (df['sigma_std'] == s_std) 
                                               & (df['sigma'] == sigma) & (df["AA_type"] == type)]
                                            
                                        metric_dict[metric] = mask[metric]
                                        ### Add mean values
                                        if not isNaN(metric_dict[metric].any()):
                                            
                                            if metric in ["NMI", "MCC"]:
                                                data_dict[a_param][b_param][k][s_std][type][k_anal][metric][sigma].append((metric_dict[metric]))
                                            elif metric in ["BDM", "loss"]:
                                                data_dict[a_param][b_param][k][s_std][type][k_anal][metric][sigma].append((metric_dict[metric]))
                                            else:
                                                data_dict[a_param][b_param][k][s_std][type][k_anal][metric][sigma].append((metric_dict[metric]))
                                        else:
                                            data_dict[a_param][b_param][k][s_std][type][k_anal][metric][sigma].append("NaN")
print("Created data dictionary succesfully")                              

#%%
######## MAKE PLOTS ######### 
# a, b, syn_k, s_std, type, anal_k, metrics, sigmas
pointScale = 120
lineScale = 3.5
def softplus(s):
    return np.log(1 + np.exp(s))

sigmas_softplussed = [softplus(s) for s in sigmas]




path = os.getcwd()
foldername = "result_plots_final"
path = os.path.join(path, foldername)

for a_param in a_params:
    for b_param in b_params:
        for k_syn in synthetic_ks:
            for s_std in sigma_stds:
                for k_anal in analysis_ks:
                    NMI_OAA, MCC_OAA, loss_OAA, sigma_est_OAA, BDM_OAA = [], [], [], [], []
                    NMI_RBOAA, MCC_RBOAA, loss_RBOAA, sigma_est_RBOAA, BDM_RBOAA = [], [], [], [], []
                    
                    NMI_CAA, MCC_CAA, loss_CAA = [], [], []
                    NMI_TSAA, MCC_TSAA, loss_TSAA = [], [], []
                    for sigma in sigmas:
                        
                        # Calc values for OAA
                        NMI_OAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["NMI"][sigma]))))
                        MCC_OAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["MCC"][sigma]))))
                        loss_OAA.append(np.min(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["loss"][sigma]))))
                        BDM_OAA.append(np.min(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["BDM"][sigma]))))
                        sigma_est_OAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["Est. sigma"][sigma]))))
                        
                         # Calc values for RBOAA
                        NMI_RBOAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["NMI"][sigma]))))
                        MCC_RBOAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["MCC"][sigma]))))
                        loss_RBOAA.append(np.min(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["loss"][sigma]))))
                        BDM_RBOAA.append(np.min(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["BDM"][sigma]))))
                        sigma_est_RBOAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["Est. sigma"][sigma]))))
                        
                        # Calc values for CAA
                        NMI_CAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["NMI"][sigma]))))
                        MCC_CAA.append(np.max(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["MCC"][sigma]))))
                        loss_CAA.append(np.min(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["loss"][sigma]))))
                        
                        # Calc values for TSAA
                        # NMI_TSAA.append(np.mean(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["NMI"][sigma]))))
                        # MCC_TSAA.append(np.mean(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["MCC"][sigma]))))
                        # loss_TSAA.append(np.min(flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["loss"][sigma]))))
                        
                        
                    
                    #### NMI PLOTS ####
                    
                    fig = plt.gcf()
                    fig.set_size_inches(10, 7)
                    plt.scatter(sigmas_softplussed, NMI_OAA, label = "OAA",s=pointScale)
                    plt.plot(sigmas_softplussed, NMI_OAA, '--', linewidth = lineScale)
                    plt.scatter(sigmas_softplussed, NMI_RBOAA, label = "RBOAA " , s=pointScale)
                    plt.plot(sigmas_softplussed, NMI_RBOAA, '--', linewidth = lineScale)
                    plt.scatter(sigmas_softplussed, NMI_CAA, label = "CAA " , s=pointScale)
                    plt.plot(sigmas_softplussed, NMI_CAA, '--', linewidth = lineScale)
                    plt.xlabel('sigma', fontsize=14)
                    plt.ylabel('NMI', fontsize=14)
                    plt.title("NMI plot OAA, RBOAA & CAA", fontsize = 20)
                    plt.legend()
                    file = ("NMI plot, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    fig.savefig(os.path.join(path,file),dpi=200)
                    plt.close()
                    
                    
                    
                    #### MCC PLOTS ####
                    
                    fig = plt.gcf()
                    fig.set_size_inches(10, 7)
                    plt.scatter(sigmas_softplussed, MCC_OAA, label = "OAA",s=pointScale)
                    plt.plot(sigmas_softplussed, MCC_OAA, '--', linewidth = lineScale)
                    plt.scatter(sigmas_softplussed, MCC_RBOAA, label = "RBOAA " , s=pointScale)
                    plt.plot(sigmas_softplussed, MCC_RBOAA, '--', linewidth = lineScale)
                    plt.scatter(sigmas_softplussed, MCC_CAA, label = "CAA " , s=pointScale)
                    plt.plot(sigmas_softplussed, MCC_CAA, '--', linewidth = lineScale)
                    plt.xlabel('sigma', fontsize=14)
                    plt.ylabel('MCC', fontsize=14)
                    plt.title("MCC plot OAA, RBOAA & CAA", fontsize = 20)
                    plt.legend()
                    file = ("MCC plot, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    fig.savefig(os.path.join(path,file),dpi=200)
                    plt.close()
                    
                    ### LOSS PLOTS ###
                    # Make loss plots for RBOAA, OAA
                    fig = plt.gcf()
                    fig.set_size_inches(10, 7)
                    plt.scatter(sigmas_softplussed, loss_OAA, label = "OAA ",s=pointScale)
                    plt.plot(sigmas_softplussed, loss_OAA, '--', linewidth = lineScale)
                    plt.scatter(sigmas_softplussed, loss_RBOAA, label = "RBOAA ",s=pointScale)
                    plt.plot(sigmas_softplussed, loss_RBOAA, '--', linewidth = lineScale)

                    plt.xlabel('sigma', fontsize=13)
                    plt.ylabel('loss', fontsize=13)
                    plt.title("Loss plot for OAA and RBOAA", fontsize = 18)
                    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits=(6,6))
                    
                    plt.legend()
                    file = ("Loss OAA+RBOAA, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    fig.savefig(os.path.join(path,file),dpi=200)
                    plt.close()  
                    
                    # Make loss plot for CAA
                    fig = plt.gcf()
                    fig.set_size_inches(10, 7)
                    plt.scatter(sigmas_softplussed, loss_CAA, label = "CAA ",s=pointScale)
                    plt.plot(sigmas_softplussed, loss_CAA, '--', linewidth = lineScale)
                    
                    plt.xlabel('sigma', fontsize=14)
                    plt.ylabel('loss', fontsize=14)
                    plt.title("Loss plot for CAA", fontsize = 22)
                    plt.legend()
                    file = ("Loss CAA, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    fig.savefig(os.path.join(path,file),dpi=200)
                    plt.close()
                    
                    
                    
                    ### SIGMA PLOTS ###
                    fig = plt.gcf()
                    fig.set_size_inches(10, 7)
                    plt.scatter(sigmas_softplussed, sigma_est_OAA, label = "OAA",s=pointScale)
                    plt.plot(sigmas_softplussed, sigma_est_OAA, '--', linewidth = lineScale)
                    plt.scatter(sigmas_softplussed, sigma_est_RBOAA, label = "RBOAA ",s=pointScale)
                    plt.plot(sigmas_softplussed, sigma_est_RBOAA, '--', linewidth = lineScale)
                    
                    plt.xlabel('true sigma', fontsize=14)
                    plt.ylabel('est. sigma', fontsize=14)
                    plt.title("Est. & true sigma for OAA & RBOAA", fontsize = 20)
                    plt.legend()
                    file = ("Sigma plot, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    fig.savefig(os.path.join(path,file),dpi=200)
                    plt.close()
                    
                    
                    ### BDM PLOTS ###
                    
                    fig = plt.gcf()
                    fig.set_size_inches(10, 7)
                    # Make BDM plot
                    plt.scatter(sigmas_softplussed, BDM_OAA, label = "OAA" ,s=pointScale)
                    plt.plot(sigmas_softplussed, BDM_OAA, '--', linewidth = lineScale)
                    plt.scatter(sigmas_softplussed, BDM_RBOAA, label = "RBOAA ",s=pointScale)
                    plt.plot(sigmas_softplussed, BDM_RBOAA, '--', linewidth = lineScale)
                    
                    plt.xlabel('sigma', fontsize=14)
                    plt.ylabel('BDM', fontsize=14)
                    plt.title("Boundary difference plot for varying sigmas", fontsize = 20)
                    plt.legend()
                    file = ("BDM plot, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    fig.savefig(os.path.join(path,file),dpi=200)
                    plt.close()
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
print("plots created and saved in folder {}".format(path))



                            
                                
                            
                            
                            
                                    
                                    
                                
                            
                                
                        
                
            
    
    
    
        




