from AAM import AA
from synthetic_data_class import _synthetic_data


lrs = [0.1, 0.01, 0.001]

AAM = AA()
AAM.create_synthetic_data(N = 40000, M=21,K=5,sigma=-20,a_param=1,b_param=1000)


#%%
for l in lrs:
    AAM.analyse(AA_type = "CAA", with_synthetic_data=True,K=5,lr=l, n_iter=25000, early_stopping=True)


    AAM.plot(plot_type="loss_plot",model_type="CAA",with_synthetic_data=True)

    # AAM.plot(plot_type = "PCA_scatter_plot", model_type = "CAA", with_synthetic_data=True)
### CAA ###
# Run1 = 57726.39, iterations = 1076, time = 30 sec
# Run2 = 57799.6, iterations = 1076, time = 28.85 sec
# Run3 = 57749, iterations = 1101, time = 29 sec
##########################################################
#%%

for l in lrs:
    AAM.analyse(AA_type = "TSAA", with_synthetic_data=True,K=5,lr=l, n_iter=25000, early_stopping=True)
    AAM.plot(plot_type="loss_plot",model_type="TSAA",with_synthetic_data=True)
### TSAA###
# Run1: Loss = 2738, time = 51 sec, iterations = 1726
# Run2: Loss = 2664, time = 108.19 sec, iterations = 1626
# Run3: Loss = 2538, time = 33.5 sec, iterations = 1201
##########################################################
#%%
for l in lrs:
    AAM.analyse(AA_type = "OAA", with_synthetic_data=True,K=5,lr=l, n_iter=25000, early_stopping=True)
    AAM.plot(plot_type="loss_plot",model_type="OAA",with_synthetic_data=True)
### OAA ###
# Run1: Loss = , time =  sec, iterations = 
# Run2: Loss = ..., time = .., iterations = ..
# Run3: Loss = x, time = .., iterations = ..
##########################################################
#%%
for l in lrs:
    AAM.analyse(AA_type = "RBOAA", with_synthetic_data=True,K=5,lr=l, n_iter=25000, early_stopping=True)
    AAM.plot(plot_type="loss_plot",model_type="RBOAA",with_synthetic_data=True)

# Run1: Loss = , time = , iterations = 
# Run2: Loss = ..., time = .., iterations = ..
# Run3: Loss = ..., time = .., iterations = ..
##########################################################





"""
      COLLECTED STATS FOR THE DIFFERENT RUNS
      
      lrs = [0.1, 0.01, 0.001]
      
      
                 ### CAA ###
                 
      # Run1 = 57726.39, iterations = 1076, time = 30 sec
      # Run2 = 57799.6, iterations = 1076, time = 28.85 sec
      # Run3 = 57749, iterations = 1101, time = 29 sec
      ##########################################################
      
      
                 ### TSAA###
                 
      # Run1: Loss = 2738, time = 51 sec, iterations = 1726
      # Run2: Loss = 2664, time = 108.19 sec, iterations = 1626
      # Run3: Loss = 2538, time = 33.5 sec, iterations = 1201
      ##########################################################
      
      
      ### OAA ###
      # Run1: Loss = , time =  sec, iterations = 
      # Run2: Loss = ..., time = .., iterations = ..
      # Run3: Loss = x, time = .., iterations = ..
      ##########################################################
      
      
      


"""