from AAM import AA
from synthetic_data_class import _synthetic_data

AAM = AA()
AAM.create_synthetic_data(N = 10000, M=21,K=5,sigma=-20,a_param=1,b_param=1000)
AAM.analyse(AA_type = "OAA", with_synthetic_data=True,K=5,lr=0.005, n_iter=25000, early_stopping=True,with_CAA_initialization=True)
AAM.plot(model_type="OAA",plot_type="loss_plot",with_synthetic_data=True)