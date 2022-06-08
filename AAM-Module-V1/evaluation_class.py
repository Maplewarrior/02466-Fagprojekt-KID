########## IMPORT ##########
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


# Matrix correlation coefficient function
def MCC( A1, A2):
        _, K1 = A1.shape 
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

def calcMI(A1, A2):
    P = A1@A2.T
    PXY = P/sum(sum(P))
    PXPY = np.outer(np.sum(PXY,axis=1), np.sum(PXY,axis=0))
    ind = np.where(PXY>0)
    MI = sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
    return MI

# Normalized mutual information function
def NMI(A1, A2):
    #krav at værdierne i række summer til 1 ???
    NMI = (2*calcMI(A1,A2)) / (calcMI(A1,A1) + calcMI(A2,A2))
    return NMI

# AA_type in ["OAA", "RBOAA"]
def _resbonse_bias_analysis(self,b_true, b_est, AA_type):
    N, J = b_true.shape
    
    if AA_type == "OAA":
        b_est = np.array([b_est[:] for _ in range(N)])
    
    return np.sum(np.abs(b_true-b_est)**2)

# def load_results(directory):
    
    
#     return motherfucker_df


def softplus(s):
    return np.log(1+np.exp(s))

def sigma_NMI_plot(NMI_result, type = "combined"):
    
    # MAKE A PLOT WITH EACH METHOD THAT MAKES SENSE
    if type == "combined":
        # print("not done")
        AA_types = list(NMI_result.keys())
        sigmas = sorted(list(NMI_result[AA_types[0]].keys()))
        archetypes = list(NMI_result[AA_types[0]][sigmas[0]].keys())
        
        a_params = list(NMI_result[AA_types[0]][sigmas[0]][archetypes[0]].keys())
        b_params = list(NMI_result[AA_types[0]][sigmas[0]][archetypes[0]][a_params[0]].keys())
        n_reps = len(list(NMI_result[AA_types[0]][sigmas[0]][archetypes[0]][a_params[0]][b_params[0]]))    
        
        colors = ["g", "black","b","r"]
        sigmas_c = [round(softplus(s),2) for s in sigmas]
        
        y_vals = np.empty((len(AA_types), len(sigmas)))
        
        # Get NMI values
        for j, type in enumerate(AA_types):
            for l, s in enumerate(sigmas):
                vals = []
                for i in range(n_reps):
                    vals.append(NMI_result[type][s][5][1][1000][i])
                # Take mean of the repetitions
                y_vals[j,l] = np.mean(vals)
        
        
        for c in range(len(AA_types)):
            plt.scatter(sigmas_c, y_vals[c], label = str(AA_types[c]))
            plt.plot(sigmas_c, y_vals[c], '--')
            plt.xlabel('sigma', fontsize=18)
            plt.ylabel('NMI', fontsize=18)
            plt.title('NMI plot for the four methods', fontsize = 22)
            
            
        plt.legend()
        plt.show()
          
    # MAKE 4 SUBPLOTS, ONE FOR EACH METHOD
    elif type == "individual":
    
        

        AA_types = list(NMI_result.keys())
        sigmas = sorted(list(NMI_result["TSAA"].keys()))
        archetypes = list(NMI_result["TSAA"][sigmas[0]].keys())
        
        a_params = list(NMI_result["TSAA"][sigmas[0]][archetypes[0]].keys())
        b_params = list(NMI_result["TSAA"][sigmas[0]][archetypes[0]][a_params[0]].keys())
        n_reps = len(list(NMI_result["TSAA"][sigmas[0]][archetypes[0]][a_params[0]][b_params[0]]))    
        
        colors = ["g", "black","b","r"]
        sigmas_c = [round(softplus(s),2) for s in sigmas]
        
        
        y_vals = np.empty((len(AA_types), len(sigmas)))
        for j, type in enumerate(AA_types):
            for l, s in enumerate(sigmas):
                vals = []
                for i in range(n_reps):
                    vals.append(NMI_result[type][s][5][1][1000][i])
                y_vals[j,l] = np.mean(vals)
        
        fig, ax = plt.subplots(2,2)
        for k, axs in enumerate(fig.axes):
            axs.scatter(sigmas_c, y_vals[k], c = colors[k])
            axs.set_title(str(AA_types[k]))
            axs.set_xlabel("sigma")
            axs.set_ylabel("NMI")
            fig.subplots_adjust(hspace=0.8, wspace=0.5)                    
                            
        plt.show()

# sigma_NMI_plot(A_NMI_results, type = "combined")




def m_cor_plot(cor_result, type = "combined"):
    
    if type == 'combined':
        AA_types = list(cor_result.keys())
        sigmas = sorted(list(cor_result["TSAA"].keys()))
        archetypes = list(cor_result["TSAA"][sigmas[0]].keys())
        
        a_params = list(cor_result["TSAA"][sigmas[0]][archetypes[0]].keys())
        b_params = list(cor_result["TSAA"][sigmas[0]][archetypes[0]][a_params[0]].keys())
        n_reps = len(list(cor_result["TSAA"][sigmas[0]][archetypes[0]][a_params[0]][b_params[0]]))
        
        
        colors = ["g", "black","b","r"]
        sigmas_c = [round(softplus(s),2) for s in sigmas]
        
        y_vals = np.empty((len(AA_types), len(sigmas)))
        for j, type in enumerate(AA_types):
            for l, s in enumerate(sigmas):
                vals = []
                for i in range(n_reps):
                    vals.append(cor_result[type][s][5][a_params[-1]][b_params[-1]][i])
                y_vals[j,l] = np.mean(vals)
        
        
        for c in range(len(AA_types)):
            plt.scatter(sigmas_c, y_vals[c], label = str(AA_types[c]))
            plt.plot(sigmas_c, y_vals[c], '--')
            plt.xlabel('sigma', fontsize=18)
            plt.ylabel('MCC', fontsize=18)
            plt.title('MCC plot for the 4 methods', fontsize = 22)
            
            
        plt.legend()
        plt.show()
            
        
    elif type == 'individual':
        
    
        AA_types = list(cor_result.keys())
        sigmas = sorted(list(cor_result["TSAA"].keys()))
        archetypes = list(cor_result["TSAA"][sigmas[0]].keys())
        
        a_params = list(cor_result["TSAA"][sigmas[0]][archetypes[0]].keys())
        b_params = list(cor_result["TSAA"][sigmas[0]][archetypes[0]][a_params[0]].keys())
        n_reps = len(list(cor_result["TSAA"][sigmas[0]][archetypes[0]][a_params[0]][b_params[0]]))
        
        
        colors = ["g", "black","b","r"]
        sigmas_c = [round(softplus(s),2) for s in sigmas]
        
        
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
            axs.set_xlabel("sigma")
            axs.set_ylabel("MCC")
            fig.subplots_adjust(hspace=0.8, wspace=0.5)
            
        
        plt.show()
    

#%%

def plot_loss(Ls):
    
    AA_types = sorted(list(Ls.keys()))
    sigmas = sorted(list(Ls[AA_types[0]].keys()))
    archetypes = sorted(list(Ls[AA_types[0]][sigmas[0]].keys()))
    a_params = sorted(list(Ls[AA_types[0]][sigmas[0]][archetypes[0]].keys()))
    b_params = sorted(list(Ls[AA_types[0]][sigmas[0]][archetypes[0]][a_params[0]].keys()))
    n_reps = len(list(Ls[AA_types[0]][sigmas[0]][archetypes[0]][a_params[0]][b_params[0]]))
    
    
    sigmas_c = [round(softplus(s),2) for s in sigmas]
    
    colors = ["g", "black","b","r"]
    
    
    L_CAA, L_TSAA, L_OAA, L_RBOAA = [], [], [], []
    vals = np.empty((len(AA_types), n_reps))
    for s in sigmas:
        for i, type in enumerate(AA_types):
            for j in range(n_reps):
                # print(len(Ls[type][s][5][a_params[1]][b_params[-1]][j]))
                vals[i,j] = Ls[type][s][5][a_params[1]][b_params[-1]][j][-1]
     
        L_CAA.append(np.mean(vals[0,:]))
        L_OAA.append(np.mean(vals[1,:]))
        L_RBOAA.append(np.mean(vals[2,:]))
        L_TSAA.append(np.mean(vals[3,:]))
    
    
    
    
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(x=sigmas_c, y=L_OAA, label = "OAA")
    ax[0].plot(sigmas_c, L_OAA)
    ax[0].scatter(x=sigmas_c, y=L_RBOAA, label = "RBOAA")
    ax[0].plot(sigmas_c, L_RBOAA)
    ax[0].set_xlabel("sigma", fontsize = 15)
    ax[0].set_ylabel("loss", fontsize = 15)
    ax[0].set_title('Loss plot for OAA and RBOAA', fontsize = 20)
    ax[0].legend()
    
    ax[1].scatter(x=sigmas_c, y=L_CAA, label = "CAA")
    ax[1].plot(sigmas_c, L_CAA)
    ax[1].scatter(x=sigmas_c, y=L_TSAA, label = "TSAA")
    ax[1].plot(sigmas_c, L_TSAA)
    ax[1].set_xlabel("sigma", fontsize = 15)
    ax[1].set_ylabel("loss", fontsize = 15)
    ax[1].set_title('Loss plot for CAA and TSAA', fontsize = 20)
    ax[1].legend()
    
    
    plt.tight_layout()
    plt.show()
    

# plot_loss(Ls)




# def create_dicts(results):
    
#     Z_NMI_results, A_NMI_results = {}, {}
#     Z_cor_results = {}
#     Ls = {}
    
#     AA_types = list(results.keys())
#     sigmas = list(results["TSAA"].keys())
    
#     archetypes = list(results["TSAA"][sigmas[0]].keys())
    
#     a_params = list(results["TSAA"][sigmas[0]][archetypes[0]].keys())
#     b_params = list(results["TSAA"][sigmas[0]][archetypes[0]][a_params[0]].keys())
    
#     n_reps = len(list(results[AA_types[0]][sigmas[0]][archetypes[0]][a_params[0]][b_params[0]]["analysis"]))
    
#     for type in AA_types:
    
#         if not type in Z_NMI_results:
            
#             Z_NMI_results[type] = {}
#             A_NMI_results[type] = {}
#             Z_cor_results[type] = {}
#             Ls[type] = {}
        
#         for s in sigmas:
#             if not s in Z_NMI_results[type]:
#                 Z_NMI_results[type][s] = {}
#                 A_NMI_results[type][s] = {}
#                 Z_cor_results[type][s] = {}
#                 Ls[type][s] = {}
                
                
                
#             for k in archetypes:
#                  if not k in Z_NMI_results[type][s]:
#                      Z_NMI_results[type][s][k] = {}
#                      A_NMI_results[type][s][k] = {}
#                      Z_cor_results[type][s][k] = {}
#                      Ls[type][s][k] = {}
                         
#                  for a in a_params:
#                      if not a in Z_NMI_results[type][s][k]:
#                          Z_NMI_results[type][s][k][a] = {}
#                          A_NMI_results[type][s][k][a] = {}
#                          Z_cor_results[type][s][k][a] = {}
#                          Ls[type][s][k][a] = {}
                         
#                      for b in b_params:
#                         if not b in Z_NMI_results[type][s][k][a]:
#                             Z_NMI_results[type][s][k][a][b] = {}
#                             A_NMI_results[type][s][k][a][b] = {}
#                             Z_cor_results[type][s][k][a][b] = {}
#                             Ls[type][s][k][a][b] = {}
                            
#                         for i in range(n_reps):
#                             if not i in Z_NMI_results[type][s][k][a][b]:
#                                 Z_NMI_results[type][s][k][a][b][i] = []
#                                 A_NMI_results[type][s][k][a][b][i] = []
#                                 Z_cor_results[type][s][k][a][b][i] = []
#                                 Ls[type][s][k][a][b][i] = []
                            
                            
#                             if type in ["CAA", "TSAA"]:
#                                 Ls[type][s][k][a][b][i] = results[type][s][k][a][b]["analysis"][i].RSS
                            
#                             else:
#                                 Ls[type][s][k][a][b][i] = results[type][s][k][a][b]["analysis"][i].loss
                            
                            
#                             A_an = results[type][s][k][a][b]["analysis"][i].A
#                             A_gt = results[type][s][k][a][b]["metadata"][i].A
#                             # print(ec._normalised_mutual_information(A_an, A_an))
                            
#                             Z_an = results[type][s][k][a][b]["analysis"][i].Z
#                             Z_gt = results[type][s][k][a][b]["metadata"][i].Z
                            
                                                
#                             A_NMI_results[type][s][k][a][b][i].append(ev._normalised_mutual_information(A_an, A_gt))
#                             Z_NMI_results[type][s][k][a][b][i].append(ev._normalised_mutual_information(Z_an.T, Z_gt.T))
                            
                            
#                             Z_cor_results[type][s][k][a][b][i] = ev._matrix_correlation_coefficient(Z_an, Z_gt)
#     return A_NMI_results, Z_NMI_results, Z_cor_results, Ls


# A_NMI_results, Z_NMI_results, Z_cor_results, Ls = create_dicts(results)



# ########## EVALUATION CLASS ##########
# class _evaluation:
    
#     def __init__(self):
#         self.results = self.load_results()
        
#         self.AA_types = list(self.results.keys())
#         self.sigmas = list(self.results[self.AA_types[0]].keys())
#         self.archetypes = list(self.results[self.AA_types[0]][self.sigmas[0]].keys())
#         self.a_params = list(self.results[self.AA_types[0]][self.sigmas[0]][self.archetypes[0]].keys())
#         self.b_params = list(self.results[self.AA_types[0]][self.sigmas[0]][self.archetypes[0]][self.a_params[0]].keys())
#         self.n_reps = len(list(self.results[self.AA_types[0]][self.sigmas[0]][self.archetypes[0]][self.a_params[0]][self.b_params[0]]["analysis"]))
        
#         self.A_NMI_results, self.Z_NMI_results, self.Z_cor_results, self.Ls = self.create_dicts(self.results)
    
    
#     def load_results(self):
#         results = {}
#         directory = 'synthetic_results'
#         for filename in os.listdir(directory):
#             filepath = os.path.join(directory, filename)
#             if os.path.isfile(filepath):
#                 file = open(filepath,'rb')
#                 result = pickle.load(file)

#                 AA_type = filename.split("_")[0]
#                 sigma = float(filename.split("_")[2])
#                 k = int(filename.split("_")[4])
#                 a = float(filename.split("_")[6])
#                 b = float(filename.split("_")[8])
                
#                 if not AA_type in results:
#                     results[AA_type] = {}
#                 if not sigma in results[AA_type]:
#                     results[AA_type][sigma] = {}
#                 if not k in results[AA_type][sigma]:
#                     results[AA_type][sigma][k] = {}
#                 if not a in results[AA_type][sigma][k]:
#                     results[AA_type][sigma][k][a] = {}
#                 if not b in results[AA_type][sigma][k][a]:
#                     results[AA_type][sigma][k][a][b] = {}
#                 if "metadata" in filename:
#                     if not "metadata" in results[AA_type][sigma][k][a][b]:
#                         results[AA_type][sigma][k][a][b]["metadata"] = []
#                     results[AA_type][sigma][k][a][b]["metadata"].append(result)
#                 elif not "metadata" in filename:
#                     if not "analysis" in results[AA_type][sigma][k][a][b]:
#                         results[AA_type][sigma][k][a][b]["analysis"] = []
#                     results[AA_type][sigma][k][a][b]["analysis"].append(result)

#         return results

#     def _matrix_correlation_coefficient(self, A1, A2):
#         _, K1 = A1.shape 
#         _, K2 = A2.shape
#         corr = np.zeros((K1,K2))
#         for i in range(K1):
#             for j in range(K2):
#                 corr[i][j] = np.corrcoef(A1[:,i], A2[:,j])[0][1]
        
#         max_list = []
#         for _ in range(min(K1,K2)):
#             row, column = np.unravel_index(corr.argmax(), corr.shape)
#             max_list.append(corr[row][column])
#             corr = np.delete(corr, row, axis=0)
#             corr = np.delete(corr, column, axis=1)
        
#         return np.mean(max_list)

#     def _calcMI(self, A1, A2):
#         P = A1@A2.T
#         PXY = P/sum(sum(P))
#         PXPY = np.outer(np.sum(PXY,axis=1), np.sum(PXY,axis=0))
#         ind = np.where(PXY>0)
#         MI = sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
#         return MI

#     def _normalised_mutual_information(self,A1,A2):
#         #krav at værdierne i række summer til 1 ???
#         NMI = (2*self._calcMI(A1,A2)) / (self._calcMI(A1,A1) + self._calcMI(A2,A2))
#         return NMI

#     # AA_type = 
#     def _resbonse_bias_analysis(self,b_true, b_est, AA_type):
#         N, J = b_true.shape
        
#         if AA_type == "OAA":
#             b_est = np.array([b_est[:] for _ in range(N)])
        
#         return np.sum(np.abs(b_true-b_est)**2)
        
            
    
    
#     def create_dicts(self, results):
    
#         Z_NMI_results, A_NMI_results = {}, {}
#         Z_cor_results = {}
#         Ls = {}
        
        
#         for type in self.AA_types:
        
#             if not type in Z_NMI_results:
                
#                 Z_NMI_results[type] = {}
#                 A_NMI_results[type] = {}
#                 Z_cor_results[type] = {}
#                 Ls[type] = {}
            
#             for s in self.sigmas:
#                 if not s in Z_NMI_results[type]:
#                     Z_NMI_results[type][s] = {}
#                     A_NMI_results[type][s] = {}
#                     Z_cor_results[type][s] = {}
#                     Ls[type][s] = {}
                    
                    
                    
#                 for k in self.archetypes:
#                      if not k in Z_NMI_results[type][s]:
#                          Z_NMI_results[type][s][k] = {}
#                          A_NMI_results[type][s][k] = {}
#                          Z_cor_results[type][s][k] = {}
#                          Ls[type][s][k] = {}
                             
#                      for a in self.a_params:
#                          if not a in Z_NMI_results[type][s][k]:
#                              Z_NMI_results[type][s][k][a] = {}
#                              A_NMI_results[type][s][k][a] = {}
#                              Z_cor_results[type][s][k][a] = {}
#                              Ls[type][s][k][a] = {}
                             
#                          for b in self.b_params:
#                             if not b in Z_NMI_results[type][s][k][a]:
#                                 Z_NMI_results[type][s][k][a][b] = {}
#                                 A_NMI_results[type][s][k][a][b] = {}
#                                 Z_cor_results[type][s][k][a][b] = {}
#                                 Ls[type][s][k][a][b] = {}
                                
#                             for i in range(self.n_reps):
#                                 if not i in Z_NMI_results[type][s][k][a][b]:
#                                     Z_NMI_results[type][s][k][a][b][i] = []
#                                     A_NMI_results[type][s][k][a][b][i] = []
#                                     Z_cor_results[type][s][k][a][b][i] = []
#                                     Ls[type][s][k][a][b][i] = []
                                
                                
#                                 if type in ["CAA", "TSAA"]:
#                                     Ls[type][s][k][a][b][i] = results[type][s][k][a][b]["analysis"][i].RSS
                                
#                                 else:
#                                     Ls[type][s][k][a][b][i] = results[type][s][k][a][b]["analysis"][i].loss
                                
                                
#                                 A_an = results[type][s][k][a][b]["analysis"][i].A
#                                 A_gt = results[type][s][k][a][b]["metadata"][i].A
#                                 # print(ec._normalised_mutual_information(A_an, A_an))
                                
#                                 Z_an = results[type][s][k][a][b]["analysis"][i].Z
#                                 Z_gt = results[type][s][k][a][b]["metadata"][i].Z
                                
                                                    
#                                 A_NMI_results[type][s][k][a][b][i].append(self._normalised_mutual_information(A_an, A_gt))
#                                 Z_NMI_results[type][s][k][a][b][i].append(self._normalised_mutual_information(Z_an.T, Z_gt.T))
                                
                                
#                                 Z_cor_results[type][s][k][a][b][i] = self._matrix_correlation_coefficient(Z_an, Z_gt)
                                
#         return A_NMI_results, Z_NMI_results, Z_cor_results, Ls

#     def softplus(self, s):
#         return np.log(1+np.exp(s))

#     #### IMPLEMENT METHODS ####
#     def sigma_NMI_plot(self, NMI_result, type = "combined"):
        
#         colors = ["g", "black","b","r"]
#         sigmas_c = [round(self.softplus(s),2) for s in self.sigmas]
#         y_vals = np.empty((len(self.AA_types), len(self.sigmas)))
        
#         # Get NMI values
#         for j, arc in enumerate(self.AA_types):
#             for l, s in enumerate(self.sigmas):
#                 vals = []
#                 for i in range(self.n_reps):
#                     vals.append(NMI_result[arc][s][5][1][1000][i])
#                 # Take mean of the repetitions
#                 y_vals[j,l] = np.mean(vals)
        
        
#         # MAKE A PLOT WITH EACH METHOD 
#         if type == "combined":
            
#             for c in range(len(self.AA_types)):
#                 plt.scatter(sigmas_c, y_vals[c], label = str(self.AA_types[c]))
#                 plt.plot(sigmas_c, y_vals[c], '--')
#                 plt.xlabel('sigma', fontsize=18)
#                 plt.ylabel('NMI', fontsize=18)
#                 plt.title('NMI plot for the four methods', fontsize = 22)
                
#             plt.legend()
#             plt.show()
              
#         # MAKE 4 SUBPLOTS, ONE FOR EACH METHOD
#         elif type == "individual":
            
#             fig, ax = plt.subplots(2,2)
#             for k, axs in enumerate(fig.axes):
#                 axs.scatter(sigmas_c, y_vals[k], c = colors[k])
#                 axs.set_title(str(self.AA_types[k]))
#                 axs.set_xlabel("sigma")
#                 axs.set_ylabel("NMI")
#                 fig.subplots_adjust(hspace=0.8, wspace=0.5)
                 
#             plt.show()
    
#     def sigma_MCC_plot(self, MCC_result, type = "combined"):
        
        
#         sigmas_c = [round(self.softplus(s),2) for s in self.sigmas]
            
#         y_vals = np.empty((len(self.AA_types), len(self.sigmas)))
#         for j, arc in enumerate(self.AA_types):
#             for l, s in enumerate(self.sigmas):
#                 vals = []
#                 for i in range(self.n_reps):
#                     vals.append(MCC_result[arc][s][5][self.a_params[-1]][self.b_params[-1]][i])
#                 y_vals[j,l] = np.mean(vals)
        
#         if type == "combined":
            
#             for c in range(len(self.AA_types)):
#                 plt.scatter(sigmas_c, y_vals[c], label = str(self.AA_types[c]))
#                 plt.plot(sigmas_c, y_vals[c], '--')
#                 plt.xlabel('sigma', fontsize=18)
#                 plt.ylabel('MCC', fontsize=18)
#                 plt.title('MCC plot for the 4 methods', fontsize = 22)
                
#             plt.legend()
#             plt.show()
            
        
#         elif type == 'individual':
            
#             colors = ["g","b","r","black"]
            
#             fig, ax = plt.subplots(2,2)
#             for k, axs in enumerate(fig.axes):
#                 axs.scatter(sigmas_c, y_vals[k], c = colors[k])
#                 axs.set_title(str(self.AA_types[k]))
#                 axs.set_xlabel("sigma")
#                 axs.set_ylabel("MCC")
#                 fig.subplots_adjust(hspace=0.8, wspace=0.5)
                
            
#             plt.show()
        
#     def loss_plot(self, Ls):
        
        
#         sigmas_c = [round(self.softplus(s),2) for s in self.sigmas]
        
#         colors = ["g", "black","b","r"]
        
        
#         L_CAA, L_TSAA, L_OAA, L_RBOAA = [], [], [], []
#         vals = np.empty((len(self.AA_types), self.n_reps))
#         for s in self.sigmas:
#             for i, type in enumerate(self.AA_types):
#                 for j in range(self.n_reps):
#                     # print(len(Ls[type][s][5][a_params[1]][b_params[-1]][j]))
#                     vals[i,j] = Ls[type][s][5][self.a_params[-1]][self.b_params[-1]][j][-1]
         
#             L_CAA.append(np.mean(vals[0,:]))
#             L_OAA.append(np.mean(vals[1,:]))
#             L_RBOAA.append(np.mean(vals[2,:]))
#             L_TSAA.append(np.mean(vals[3,:]))
        
        
        
#         fig, ax = plt.subplots(1,2)
#         ax[0].scatter(x=sigmas_c, y=L_OAA, label = "OAA")
#         ax[0].plot(sigmas_c, L_OAA)
#         ax[0].scatter(x=sigmas_c, y=L_RBOAA, label = "RBOAA")
#         ax[0].plot(sigmas_c, L_RBOAA)
#         ax[0].set_xlabel("sigma", fontsize = 14)
#         ax[0].set_ylabel("loss", fontsize = 14)
#         ax[0].set_title('Loss plot for OAA and RBOAA', fontsize = 16)
#         ax[0].legend()
        
#         ax[1].scatter(x=sigmas_c, y=L_CAA, label = "CAA")
#         ax[1].plot(sigmas_c, L_CAA)
#         ax[1].scatter(x=sigmas_c, y=L_TSAA, label = "TSAA")
#         ax[1].plot(sigmas_c, L_TSAA)
#         ax[1].set_xlabel("sigma", fontsize = 14)
#         ax[1].set_ylabel("loss", fontsize = 14)
#         ax[1].set_title('Loss plot for CAA and TSAA', fontsize = 16)
#         ax[1].legend()
        
        
#         plt.tight_layout()
#         plt.show()