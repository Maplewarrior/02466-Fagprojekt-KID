from webbrowser import Galeon
from AAM import AA
from synthetic_data_class import _synthetic_data
import numpy as np
from eval_measures import NMI
from eval_measures import MCC
from RBOAA_classOLDVERSION import _RBOAA
from plots_class import _plots


AAM = AA()

#AAM.load_csv("ESS8_data.csv",range(12,33), rows = 1000)

AAM.create_synthetic_data(N = 1000, M=5,K=2,sigma=-20,a_param=1,b_param=1000, rb=False,p=6)

AAM.analyse(AA_type = "OAA", with_synthetic_data=True, K=2,lr=0.01, n_iter=5000, early_stopping=False,with_CAA_initialization=False, p=6)
# AAM.plot(model_type="RBOAA",plot_type="loss_plot", with_synthetic_data=True)
AAM.plot(model_type="OAA",plot_type="barplot_all", with_synthetic_data=True)
# AAM.plot(model_type="RBOAA",plot_type="PCA_scatter_plot", with_synthetic_data=True)

analysis_A = AAM._synthetic_results["OAA"][0].A
analysis_Z = AAM._synthetic_results["OAA"][0].Z
analysis_beta = AAM._synthetic_results["OAA"][0].b
syn_A = AAM._synthetic_data.A
syn_Z = AAM._synthetic_data.Z
syn_beta = AAM._synthetic_data.betas


print("ZZZZZZ TRUE")
print(syn_Z)
print("ZZZZZZ ANALYSIS")
print(analysis_Z)

print("AAAAAAA TRUE")
print(syn_A)
print("AAAAAAA ANALYSIS")
print(analysis_A)

print("BETAS TRUE")
print(syn_beta)
print("BETAS ANALYSIS")
print(analysis_beta)


plotter = _plots()

plotter._barplot_all(syn_Z, AAM._synthetic_results["OAA"][0].columns)

# print("SIGMA TRUE")
# print(-2)

