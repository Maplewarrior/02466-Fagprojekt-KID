from AAM import AA
from synthetic_data_class import _synthetic_data

AAM = AA()
AAM.create_synthetic_data(N = 10000, M=21,K=5,sigma=-20,a_param=1,b_param=1000)
AAM.analyse(AA_type = "RBOAA", with_synthetic_data=True,K=5, n_iter=35000, early_stopping=True)

AAM.plot(plot_type="loss_plot",model_type="RBOAA",with_synthetic_data=True)