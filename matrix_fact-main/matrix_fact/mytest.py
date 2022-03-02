import numpy as np
import pandas as pd
from aa import AA

'''
data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])

aa_mdl = AA(data, num_bases=2)
aa_mdl.factorize(niter=5)

data = np.array([[1.5], [1.2]])
W = np.array([[1.0, 0.0], [0.0, 1.0]])
a_mdl = AA(data, num_bases=2)
aa_mdl.W = W
aa_mdl.factorize(niter=5, compute_w=False)
'''
print('')

df = pd.read_csv('ESS8_data.csv')
X = df[['SD1', 'PO1', 'UN1', 'AC1', 'SC1',
       'ST1', 'CO1', 'TR1', 'HD1', 'AC2', 'SC2', 'ST2',
       'CO2', 'PO2', 'BE2', 'TR2', 'HD2']].iloc[range(100),:]
X = X.to_numpy().T
N, M = X.T.shape
K = 9

