from AAM import AA


#types={"Openness To Change": [0,10,5,14],"Self-Enhancement": [3,12,1,16],"Conservation": [4,13,6,15],"Self-Trancendence":[2,7,11,17]}

AAM = AA()
AAM.load_csv("ESS8_data.csv", columns=list(range(12, 32+1)), rows=None)
AAM.analyse(K=3, n_iter=1000, AA_type="CAA", lr=0.001, mute=False)
AAM.plot("CAA", plot_type="mixture_plot")
AAM.plot("CAA", plot_type = "PCA_scatter_plot")

# #%%
# AAM.load_analysis(model_type = "CAA")
# AAM.load_analysis(model_type = "OAA")

# AAM.plot("CAA", plot_type="barplot",archetype_number=0)
# AAM.plot("OAA", plot_type="barplot",archetype_number=0)
# AAM.plot("OAA", plot_type="mixture_plot")


# #%%

# AAM.load_csv("ESS8_data.csv", columns=list(range(12, 32+1)), rows=None)
# AAM.analyse(K=5, n_iter=1000, AA_type="TSAA", lr=0.001, mute=False)
# AAM.save_analysis(model_type="TSAA")

# #%%
# AAM = AA()
# AAM.load_analysis(model_type="TSAA")
# AAM.plot(model_type="TSAA", plot_type="PCA_scatter_plot")
# AAM.plot(model_type="TSAA", plot_type="mixture_plot")
# AAM.plot(model_type="TSAA", plot_type="barplot", archetype_number=0)
# AAM.plot(model_type="TSAA", plot_type="barplot_all")
# AAM.plot(model_type="TSAA", plot_type="loss_plot")
# AAM.plot(model_type="TSAA", plot_type="attribute_scatter_plot", attributes = [1,2])

# #%%
# #### Testing the Two-step AA method ####

# AAM.load_csv("ESS8_data.csv", columns=list(range(12, 32+1)), rows=None)
# AAM.analyse(K=5, n_iter=1000, AA_type="TSAA", lr=0.001, mute=False)


# AAM.plot("TSAA", plot_type="PCA_scatter_plot")
# AAM.plot("TSAA", plot_type="mixture_plot")

