from AAM import AA

AAM = AA()
""" AAM2 = AA()
AAM.synthetic_data(M=15, N=1000, K=5, p=6)
AAM2.synthetic_data(M=15, N=1000, K=5, p=6)
AAM.analyse(K=5, AA_type="OAA")
AAM2.analyse(K=5, AA_type="TSAA")
AAM.plot(model_type="OAA", plot_type= "mixture_plot")
AAM2.plot(model_type="TSAA", plot_type= "mixture_plot") """

"""
Problem: All values in X_synthetic are unique. 
This is due to X being computed as Z@A where A is found by sampling from a Dirichlet distribution.
This messes with TSAA. 
Presumably, this
"""


AAM.load_csv("ESS8_data.csv",range(12,33))
AAM.analyse(AA_type="OAA", n_iter=50)
AAM.plot(model_type="OAA", plot_type="PCA_scatter_plot")