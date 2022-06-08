from AAM import AA
from synthetic_data_class import _synthetic_data


lrs = [0.1, 0.01, 0.001]

AAM = AA()
AAM.create_synthetic_data(N = 10000, M=21,K=5,sigma=-20,a_param=1,b_param=1000)

results = {}

AAM.analyse(AA_type = "OAA", with_synthetic_data=True,K=5,lr=l, n_iter=25000, early_stopping=True)