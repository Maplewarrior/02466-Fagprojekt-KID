from AAM import AA
from synthetic_data_class import _synthetic_data

AAM = AA()
# AAM.load_csv("ESS8_data.csv",range(12,33), rows = 1000)
# AAM.analyse(AA_type = "OAA")
# AAM.plot(model_type="OAA")

#AAM.create_synthetic_data(N = 20000, M=10,K=3)
#AAM.analyse(AA_type = "OAA", with_synthetic_data=True,K=3)
#AAM.plot(model_type = "OAA", plot_type = "loss_plot", with_synthetic_data=True)

AAM.load_analysis(with_synthetic_data=True, model_type="RBOAA", filename="_sigma_1_Arche_5")
AAM.plot(model_type="RBOAA", with_synthetic_data=True, plot_type="attribute_scatter_plot")
AAM.plot(model_type="RBOAA", with_synthetic_data=True, plot_type="loss_plot")
#print(AAM._synthetic_data.X)


# Generate matrix for evaluation tests
"""np.random.seed(42)
K, N = 4, 5
A1 = np.random.rand(K,N)
A2 = np.random.rand(K,N)
"""


