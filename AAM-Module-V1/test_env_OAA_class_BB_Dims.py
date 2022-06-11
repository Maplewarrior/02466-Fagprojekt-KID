### TESTING OAA_class_BB_Dims ###
from synthetic_data_class import _synthetic_data
from OAA_class_BB_Dims import _OAA
rb = False
betaParm = 1000
K = 4
N = 10000
M = 21
epokes = 1000
sigma = -4.6
p = 5
a_param = 1



data = _synthetic_data(N=N, M=M ,K=K, p=p, sigma=sigma, rb=rb, a_param = a_param, b_param = betaParm)
# Make synthetic data be 0, ... len(p)-1
X = (data.X).T - 1



A_true = (data.A).T
Z_true = (data.Z).T

OAA = _OAA()
result = OAA._compute_archetypes(X=X, K=K, p=p, n_iter=1000, lr=0.01, mute=False, columns=data.columns, with_synthetic_data = True, early_stopping = True, with_CAA_initialization = False)
result._plot("loss_plot",["1","2","3","4","5"],1,1,1)
