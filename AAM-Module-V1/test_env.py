from AAM import AA


#types={"Openness To Change": [0,10,5,14],"Self-Enhancement": [3,12,1,16],"Conservation": [4,13,6,15],"Self-Trancendence":[2,7,11,17]}

AAM = AA()

AAM.load_analysis(model_type = "CAA")
AAM.load_analysis(model_type = "OAA")

AAM.plot("CAA", plot_type="barplot",archetype_number=0)
AAM.plot("OAA", plot_type="barplot",archetype_number=0)



#%%
#### Testing the Two-step AA method ####

AAM.load_csv("ESS8_data.csv", columns=list(range(12, 32+1)), rows=None)
AAM.analyse(K=3, n_iter=1000, AA_type="TSAA", lr=0.001, mute=False)


AAM.plot("TSAA", plot_type="PCA_scatter_plot")
AAM.plot("TSAA", plot_type="mixture_plot")

