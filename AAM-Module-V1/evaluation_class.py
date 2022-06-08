########## IMPORT ##########
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

########## EVALUATION CLASS ##########
class _evaluation:

    def load_results(self):
        results = {}
        directory = 'synthetic_results_template'
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file = open(filepath,'rb')
                result = pickle.load(file)

                AA_type = filename.split("_")[0]
                sigma = float(filename.split("_")[2])
                k = int(filename.split("_")[4])
                a = float(filename.split("_")[6])
                b = float(filename.split("_")[8])
                
                if not AA_type in results:
                    results[AA_type] = {}
                if not sigma in results[AA_type]:
                    results[AA_type][sigma] = {}
                if not k in results[AA_type][sigma]:
                    results[AA_type][sigma][k] = {}
                if not a in results[AA_type][sigma][k]:
                    results[AA_type][sigma][k][a] = {}
                if not b in results[AA_type][sigma][k][a]:
                    results[AA_type][sigma][k][a][b] = {}
                if "metadata" in filename:
                    if not "metadata" in results[AA_type][sigma][k][a][b]:
                        results[AA_type][sigma][k][a][b]["metadata"] = []
                    results[AA_type][sigma][k][a][b]["metadata"].append(result)
                elif not "metadata" in filename:
                    if not "analysis" in results[AA_type][sigma][k][a][b]:
                        results[AA_type][sigma][k][a][b]["analysis"] = []
                    results[AA_type][sigma][k][a][b]["analysis"].append(result)

        return results

    def _matrix_correlation_coefficient(self, A1, A2):
        _,K1 = A1.shape
        _, K2 = A2.shape 
        corr = np.zeros((K1,K2))
        for i in range(K1):
            for j in range(K2):
                corr[i][j] = np.corrcoef(A1[:,i], A2[:,j])[0][1]
        
        max_list = []
        for _ in range(min(K1,K2)):
            row, column = np.unravel_index(corr.argmax(), corr.shape)
            max_list.append(corr[row][column])
            corr = np.delete(corr, row, axis=0)
            corr = np.delete(corr, column, axis=1)
        
        return np.mean(max_list)

    def _calcMI(self, A1, A2):
        P = A1@A2.T
        PXY = P/sum(sum(P))
        PXPY = np.outer(np.sum(PXY,axis=1), np.sum(PXY,axis=0))
        ind = np.where(PXY>0)
        MI = sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
        return MI

    def _normalised_mutual_information(self,A1,A2):
        #krav at værdierne i række summer til 1 ???
        NMI = (2*self._calcMI(A1,A2)) / (self._calcMI(A1,A1) + self._calcMI(A2,A2))
        return NMI

    def _resbonse_bias_analysis(self,b1,b2):
        total = 0
        k = b1.shape[0]
        for i in range(k):
            total += abs(b1[i]-b2[i])
        return total/k




ec = _evaluation()
results = ec.load_results()


#%%
AA_types = list(results.keys())
sigmas = list(results["TSAA"].keys())

archetypes = list(results["TSAA"][sigmas[0]].keys())

a_params = list(results["TSAA"][sigmas[0]][archetypes[0]].keys())
b_params = list(results["TSAA"][sigmas[0]][archetypes[0]][a_params[0]].keys())

n_reps = len(list(results[AA_types[0]][sigmas[0]][archetypes[0]][a_params[0]][b_params[0]]))
Z_NMI_results, A_NMI_results = {}, {}
cor_results = {}




for type in AA_types:
    
    if not type in Z_NMI_results:
        
        Z_NMI_results[type] = {}
        A_NMI_results[type] = {}
        cor_results[type] = {}
    
    for s in sigmas:
        if not s in Z_NMI_results[type]:
            Z_NMI_results[type][s] = {}
            A_NMI_results[type][s] = {}
            cor_results[type][s] = {}
            
            
            
        for k in archetypes:
             if not k in Z_NMI_results[type][s]:
                 Z_NMI_results[type][s][k] = {}
                 A_NMI_results[type][s][k] = {}
                 cor_results[type][s][k] = {}
                     
             for a in a_params:
                 if not a in Z_NMI_results[type][s][k]:
                     Z_NMI_results[type][s][k][a] = {}
                     A_NMI_results[type][s][k][a] = {}
                     cor_results[type][s][k][a] = {}
                     
                 for b in b_params:
                    if not b in Z_NMI_results[type][s][k][a]:
                        Z_NMI_results[type][s][k][a][b] = {}
                        A_NMI_results[type][s][k][a][b] = {}
                        cor_results[type][s][k][a][b] = {}
                        
                    for i in range(n_reps):
                        if not i in Z_NMI_results[type][s][k][a][b]:
                            Z_NMI_results[type][s][k][a][b][i] = []
                            A_NMI_results[type][s][k][a][b][i] = []
                            cor_results[type][s][k][a][b][i] = []
                                
                        A_an = results[type][s][k][a][b]["analysis"][i].A
                        A_gt = results[type][s][k][a][b]["metadata"][i].A
                        # print(ec._normalised_mutual_information(A_an, A_an))
                        
                        Z_an = results[type][s][k][a][b]["analysis"][i].Z
                        Z_gt = results[type][s][k][a][b]["metadata"][i].Z
                        
                        
                        # print(A_an.shape, A_gt.shape)
                        # # print("For Z: ", Z_an.shape, Z_gt.shape)
                        
                        # print((A_an @ A_gt.T).shape)
                        A_NMI_results[type][s][k][a][b][i].append(ec._normalised_mutual_information(A_an, A_gt))
                        Z_NMI_results[type][s][k][a][b][i].append(ec._normalised_mutual_information(Z_an.T, Z_gt.T))
                        
                        print(Z_an.shape, Z_gt.shape)
                        cor_results[type][s][k][a][b] = ec._matrix_correlation_coefficient(Z_an, Z_gt)
                    
                    



#%%
print(A_an.shape, A_gt.shape)
M1 = A_an @ A_gt.T
print(M1.shape)

M2 = Z_an.T @ Z_gt
print(Z_gt.shape, Z_an.shape)
print(M2.shape)
# sigmas = list(results["CAA"].keys())
# sigmas = sorted(sigmas)
# print(sigmas)
# n_reps = 10
# y_vals, vals = [], []
# a_params = list(results["CAA"][-0.43][3].keys())
# b_params = list(results["CAA"][-0.43][3][0.85].keys())

# print(a_params)
# print(b_params)
#%%
# for s in sigmas:
#     for i in range(n_reps):
#         vals.append(NMI_results["OAA"][s][5][a_params[0]][b_params[0]][i])
#     y_vals.append(np.mean(vals))
        
# print(y_vals)
# print(vals)


#%%


def softplus(s):
    return np.log(1+np.exp(s))

def sigma_NMI_plot(NMI_result):
    
    AA_types = list(results.keys())
    sigmas = sorted(list(results["TSAA"].keys()))
    archetypes = list(results["TSAA"][sigmas[0]].keys())
    
    a_params = list(results["TSAA"][sigmas[0]][archetypes[0]].keys())
    b_params = list(results["TSAA"][sigmas[0]][archetypes[0]][a_params[0]].keys())
    n_reps = 1
    print(a_params)
    print(b_params)
    
    
    
    
    colors = ["g", "black","b","r"]
    sigmas_c = [round(softplus(s),2) for s in sigmas]
    
    
    y_vals = np.empty((len(AA_types), len(sigmas)))
    for j, type in enumerate(AA_types):
        for l, s in enumerate(sigmas):
            vals = []
            for i in range(n_reps):
                vals.append(NMI_results[type][s][5][1][1000][i])
            y_vals[j,l] = np.mean(vals)
    
    fig, ax = plt.subplots(2,2)
    for k, axs in enumerate(fig.axes):
        axs.scatter(sigmas_c, y_vals[k], c = colors[k])
        axs.set_title(str(AA_types[k]))
        fig.subplots_adjust(hspace=1, wspace=1)                    
                        
    plt.show()

sigma_NMI_plot(NMI_results)

#%%

def m_cor_plot(cor_result):
    
    colors = ["g", "black","b","r"]
    AA_types = list(results.keys())
    sigmas = list(results["CAA"].keys())
    sigmas_c = [softplus(s) for s in sigmas]
    archetypes = list(results["CAA"][-0.43].keys())
    a_params = list(results["CAA"][-0.43][3].keys())
    b_params = list(results["CAA"][-0.43][3][0.85].keys())
    n_reps = 10
    
    
    y_vals = np.empty((len(AA_types), len(sigmas)))
    for j, type in enumerate(AA_types):
        for l, s in enumerate(sigmas):
            vals = []
            for i in range(n_reps):
                vals.append(cor_result[type][s][5][a_params[-1]][b_params[-1]][i])
            y_vals[j,l] = np.mean(vals)
    
    fig, ax = plt.subplots(2,2)
    for k, axs in enumerate(fig.axes):
        axs.scatter(sigmas_c, y_vals[k], c = colors[k])
        axs.set_title(str(AA_types[k]))
        fig.subplots_adjust(hspace=1, wspace=1)
        
    
    plt.show()
# m_cor_plot(cor_results())
#%%

    





        
        
            
            
        
        