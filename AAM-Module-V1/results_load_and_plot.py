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
                                        
                                        for rep in range(n_reps):
                                            
                                            mask = df.loc[(df['a_param'] == a_param) & (df['b_param'] == b_param) 
                                                   & (df['synthetic_k'] == k) & (df['sigma_std'] == s_std) 
                                                   & (df['sigma'] == sigma) & (df['rep'] == rep) & (df["AA_type"] == type)]
                                            
                                            metric_dict[metric].append(mask[metric])
                                        
                                        ### Add mean values
                                        if not isNaN(metric_dict[metric]):
                                            
                                            data_dict[a_param][b_param][k][s_std][type][k_anal][metric][sigma].append(np.mean(metric_dict[metric]))
                                        
                                        else:
                                            data_dict[a_param][b_param][k][s_std][type][k_anal][metric][sigma].append("NaN")
                                    
#%% 
# print(df['a_param'][0])
# print(df['b_param'][0])
# print(df['synthetic_k'][0])
# print(df['sigma_std'][0])
# print(df['AA_type'][0])
# print(df['analysis_k'][0])
# print(df['sigma'][0])
# print(df['rep'][0])
# print(df["NMI"][0])

# print(data_dict[2.0]["RB_false"][5.0][0.0]["CAA"][5.0]["NMI"][-20.0])
#%%
######## MAKE PLOTS ######### 
# a, b, syn_k, s_std, type, anal_k, metrics, sigmas

def softplus(s):
    return np.log(1 + np.exp(s))

sigmas_softplussed = [softplus(s) for s in sigmas]

folder1 = "Plots_syn_K = anal_K"
folder2 = "Plots_syn_K != anal_K"



# print(flatten(list(data_dict[a_params[0]][b_params[0]][synthetic_ks[0]][sigma_stds[0]][AA_types[0]][analysis_ks[0]][metrics[0]].values())))

# print(data_dict[a_params[0]][b_params[0]][synthetic_ks[0]])


path = os.getcwd()
foldername = "result_plots_final"
path = os.path.join(path, foldername)

for a_param in a_params:
    for b_param in b_params:
        for k_syn in synthetic_ks:
            for s_std in sigma_stds:
                for k_anal in analysis_ks:
                    # for sigma in sigmas:
                    
                    # Make combined NMI and MCC plots for OAA and RBOAA
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["NMI"].values())), label = "OAA " + "NMI")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["NMI"].values())), '--')
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["NMI"].values())), label = "RBOAA " + "NMI")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["NMI"].values())), '--')
                    
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["MCC"].values())), label = "OAA " + "MCC")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["MCC"].values())), '--')
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["MCC"].values())), label = "RBOAA " + "MCC")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["MCC"].values())), '--')
                    
                    plt.xlabel('sigma', fontsize=18)
                    plt.ylabel('NMI and MCC', fontsize=18)
                    plt.title("NMI and MCC plot OAA & RBOAA", fontsize = 20)
                    plt.legend()
                    file = ("OAA+RBOAA NMI & MCC,a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    plt.savefig(os.path.join(path,file))
                    plt.close()

                    # Make combined NMI and MCC plots for CAA and TSAA
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["NMI"].values())), label = "CAA " + "NMI")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["NMI"].values())), '--')
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["NMI"].values())), label = "TSAA " + "NMI")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["NMI"].values())), '--')
                    
                    
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["MCC"].values())), label = "CAA " + "MCC")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["MCC"].values())), '--')
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["MCC"].values())), label = "TSAA " + "MCC")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["MCC"].values())), '--')
                    
                    plt.xlabel('sigma', fontsize=18)
                    plt.ylabel('NMI and MCC', fontsize=18)
                    plt.title("NMI and MCC plot CAA & TSAA", fontsize = 20)
                    plt.legend()
                    file = ("NMI+MCC,a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    plt.savefig(os.path.join(path,file))
                    plt.close()                        

                    # Make loss plots for RBOAA, OAA
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["loss"].values())), label = "OAA ")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["OAA"][k_anal]["loss"].values())), '--')
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["loss"].values())), label = "RBOAA ")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["RBOAA"][k_anal]["loss"].values())), '--')

                    plt.xlabel('sigma', fontsize=18)
                    plt.ylabel('loss', fontsize=18)
                    plt.title("Loss plot for OAA and RBOAA", fontsize = 22)
                    plt.legend()
                    file = ("Loss OAA+RBOAA, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    plt.savefig(os.path.join(path,file))
                    plt.close()
                    
                    
                    # Make loss plots for CAA and TSAA
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["loss"].values())), label = "CAA ")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["CAA"][k_anal]["loss"].values())), '--')
                    plt.scatter(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["loss"].values())), label = "TSAA ")
                    plt.plot(sigmas_softplussed, flatten(list(data_dict[a_param][b_param][k_syn][s_std]["TSAA"][k_anal]["loss"].values())), '--')

                    plt.xlabel('sigma', fontsize=18)
                    plt.ylabel('loss', fontsize=18)

                    plt.title("Loss plot for CAA and TSAA", fontsize = 22)
                    plt.legend()
                    file = ("Loss CAA+TSAA, a={0}, b={1}, k_syn={2}, k_anal={3}, s_std={4}.png".format(a_param, b_param, k_syn, k_anal, s_std))
                    plt.savefig(os.path.join(path,file))
                    plt.close()
                    
                        
                        
                        


                        
                        
                        
                        
                    
                    
                    
                        
                        
                        
                    
                    
        
        
   
                            
                                
                            
                            
                            
                                    
                                    
                                
                            
                                
                        
                
            
    
    
    
        




